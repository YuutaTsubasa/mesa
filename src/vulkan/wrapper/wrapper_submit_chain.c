/*
 * WRAPPER_CHAIN_SUBMITS=1: make the GPU run queue submissions strictly one
 * after another.
 *
 * Every batch waits for the previous batch -- on any queue -- through one
 * device-wide timeline semaphore and signals its own completion on it.  The
 * CPU never waits; only the GPU-side overlap between batches goes away.
 *
 * This works around the Mali proprietary driver on 5th-gen (CSF) GPUs such as
 * the Mali-G1-Ultra, which starts a batch before the previous batch on the
 * same queue has finished writing, despite the full barrier vkd3d-proton
 * records at the start of each command list (Sonic Racing CrossWorlds: GPU
 * page faults -- GROUP_ERROR_FATAL, MMU translation fault -- black models,
 * flickering, then the game exits; VKD3D_CONFIG=single_queue does not help,
 * chaining does).  WRAPPER_FAULT_HUNT_WAIT=2 turns the same chaining on.
 */

#include "wrapper_private.h"
#include "wrapper_log.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/simple_mtx.h"
#include "vk_util.h"

static int chain_enabled = -1;
static int serialize_enabled = -1;
static unsigned serialize_delay_us;
static simple_mtx_t chain_mtx = SIMPLE_MTX_INITIALIZER;

/* WRAPPER_SERIALIZE_SUBMITS=1: after each submission the CPU waits until the
 * GPU has finished it (plus WRAPPER_SERIALIZE_DELAY_US) before anything else
 * is submitted or the call returns.  Unlike chaining, this also keeps the
 * application from reusing memory the GPU may still be writing when the
 * driver reports completion early. */
static int fill_enabled = -1;
static simple_mtx_t fill_mtx = SIMPLE_MTX_INITIALIZER;

/* WRAPPER_FILL_EMPTY_SUBMITS=1: a batch with no command buffers (a bare
 * semaphore/fence signal, which is how vkd3d-proton implements
 * ID3D12CommandQueue::Signal and its own fence waits) gets one wrapper-owned
 * command buffer holding a full memory barrier.  Tests whether the Mali
 * driver signals empty batches before earlier work on the queue has
 * finished writing, which would let the application reuse memory the GPU
 * is still using. */
VkCommandBuffer
wrapper_fill_cb(struct wrapper_queue *queue)
{
   if (fill_enabled == -1) {
      const char *e = getenv("WRAPPER_FILL_EMPTY_SUBMITS");
      fill_enabled = e && atoi(e);
      if (fill_enabled)
         WRAPPER_LOG(info, "WRAPPER_FILL_EMPTY_SUBMITS: empty batches get a barrier");
   }
   uint32_t family = queue->vk.queue_family_index;
   if (!fill_enabled || family >= 8)
      return VK_NULL_HANDLE;

   struct wrapper_device *device = queue->device;
   simple_mtx_lock(&fill_mtx);
   if (!device->fill_cb[family]) {
      VkCommandPoolCreateInfo cpci = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .queueFamilyIndex = family,
      };
      VkCommandBufferAllocateInfo cbai = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      VkCommandBufferBeginInfo bi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      };
      VkMemoryBarrier barrier = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      };
      VkCommandPool pool = VK_NULL_HANDLE;
      VkCommandBuffer cb = VK_NULL_HANDLE;
      if (device->dispatch_table.CreateCommandPool(device->dispatch_handle, &cpci, NULL,
                                                   &pool) == VK_SUCCESS &&
          (cbai.commandPool = pool,
           device->dispatch_table.AllocateCommandBuffers(device->dispatch_handle, &cbai,
                                                         &cb) == VK_SUCCESS) &&
          device->dispatch_table.BeginCommandBuffer(cb, &bi) == VK_SUCCESS) {
         device->dispatch_table.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                                   1, &barrier, 0, NULL, 0, NULL);
         if (device->dispatch_table.EndCommandBuffer(cb) == VK_SUCCESS) {
            device->fill_pool[family] = pool;
            device->fill_cb[family] = cb;
            pool = VK_NULL_HANDLE;
         }
      }
      if (pool)
         device->dispatch_table.DestroyCommandPool(device->dispatch_handle, pool, NULL);
   }
   VkCommandBuffer cb = device->fill_cb[family];
   simple_mtx_unlock(&fill_mtx);
   return cb;
}

static bool
serialize_on(void)
{
   if (serialize_enabled == -1) {
      const char *e = getenv("WRAPPER_SERIALIZE_SUBMITS");
      serialize_enabled = e && atoi(e);
      const char *d = getenv("WRAPPER_SERIALIZE_DELAY_US");
      serialize_delay_us = d && atoi(d) > 0 ? atoi(d) : 0;
   }
   return serialize_enabled;
}

bool
wrapper_submit_chain_enabled(void)
{
   if (chain_enabled == -1) {
      const char *e = getenv("WRAPPER_CHAIN_SUBMITS");
      chain_enabled = (e && atoi(e)) || wrapper_fault_hunt_wait_mode() == 2;
   }
   return chain_enabled;
}

void
wrapper_submit_chain_lock(void)
{
   if (wrapper_submit_chain_enabled() || serialize_on())
      simple_mtx_lock(&chain_mtx);
}

void
wrapper_submit_chain_unlock(struct wrapper_queue *queue, VkResult result)
{
   if (serialize_on() && result == VK_SUCCESS) {
      struct wrapper_device *device = queue->device;
      VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
      VkFence fence = VK_NULL_HANDLE;
      if (device->dispatch_table.CreateFence(device->dispatch_handle, &fci, NULL, &fence) == VK_SUCCESS) {
         VkCommandBuffer fill = wrapper_fill_cb(queue);
         VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &fill,
         };
         if (device->dispatch_table.QueueSubmit(queue->dispatch_handle, fill ? 1 : 0,
                                                fill ? &si : NULL, fence) == VK_SUCCESS)
            device->dispatch_table.WaitForFences(device->dispatch_handle, 1, &fence, VK_TRUE,
                                                 UINT64_MAX);
         device->dispatch_table.DestroyFence(device->dispatch_handle, fence, NULL);
      }
      if (serialize_delay_us)
         usleep(serialize_delay_us);
   }
   if (wrapper_submit_chain_enabled() || serialize_on())
      simple_mtx_unlock(&chain_mtx);
}

static bool
chain_semaphore(struct wrapper_device *device)
{
   if (device->hunt_chain)
      return true;
   VkSemaphoreTypeCreateInfo type = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
   };
   VkSemaphoreCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type,
   };
   device->hunt_chain_value = 0;
   return device->dispatch_table.CreateSemaphore(device->dispatch_handle, &sci, NULL,
                                                 &device->hunt_chain) == VK_SUCCESS;
}

/* vkQueueSubmit2: values travel inline.  Called with the chain lock held.
 * Returns memory to free after the submit, or NULL. */
void *
wrapper_submit_chain2(struct wrapper_device *device, VkSubmitInfo2 *submit)
{
   if (!wrapper_submit_chain_enabled() || !chain_semaphore(device))
      return NULL;

   uint32_t waits = submit->waitSemaphoreInfoCount, signals = submit->signalSemaphoreInfoCount;
   VkSemaphoreSubmitInfo *mem = malloc((waits + signals + 2) * sizeof(*mem));
   if (!mem)
      return NULL;
   VkSemaphoreSubmitInfo *w = mem, *sg = mem + waits + 1;
   memcpy(w, submit->pWaitSemaphoreInfos, waits * sizeof(*w));
   memcpy(sg, submit->pSignalSemaphoreInfos, signals * sizeof(*sg));
   w[waits] = (VkSemaphoreSubmitInfo) {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = device->hunt_chain,
      .value = device->hunt_chain_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   sg[signals] = (VkSemaphoreSubmitInfo) {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = device->hunt_chain,
      .value = ++device->hunt_chain_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   submit->pWaitSemaphoreInfos = w;
   submit->waitSemaphoreInfoCount = waits + 1;
   submit->pSignalSemaphoreInfos = sg;
   submit->signalSemaphoreInfoCount = signals + 1;
   return mem;
}

/* Legacy vkQueueSubmit: timeline values live in VkTimelineSemaphoreSubmitInfo,
 * whose arrays must cover every semaphore of the batch.  Ours replaces the
 * application's when that one heads the pNext chain (vkd3d-proton's case) or
 * is prepended when there is none; anything else is left unchained. */
void *
wrapper_submit_chain1(struct wrapper_device *device, VkSubmitInfo *submit)
{
   if (!wrapper_submit_chain_enabled() || !chain_semaphore(device))
      return NULL;

   const VkTimelineSemaphoreSubmitInfo *app_ti =
      vk_find_struct_const(submit->pNext, TIMELINE_SEMAPHORE_SUBMIT_INFO);
   const void *rest = submit->pNext;
   if (app_ti) {
      if ((const void *)app_ti != submit->pNext) {
         static bool warned;
         if (!warned) {
            warned = true;
            WRAPPER_LOG(error, "WRAPPER_CHAIN_SUBMITS: timeline info not first in pNext, batch left unchained");
         }
         return NULL;
      }
      rest = app_ti->pNext;
   }

   uint32_t waits = submit->waitSemaphoreCount, signals = submit->signalSemaphoreCount;
   size_t size = sizeof(VkTimelineSemaphoreSubmitInfo) +
                 (waits + 1) * (sizeof(VkSemaphore) + sizeof(VkPipelineStageFlags) + sizeof(uint64_t)) +
                 (signals + 1) * (sizeof(VkSemaphore) + sizeof(uint64_t));
   uint8_t *mem = calloc(1, size);
   if (!mem)
      return NULL;

   VkTimelineSemaphoreSubmitInfo *ti = (void *)mem;
   uint64_t *wv = (uint64_t *)(ti + 1);
   uint64_t *sv = wv + waits + 1;
   VkSemaphore *ws = (VkSemaphore *)(sv + signals + 1);
   VkSemaphore *ss = ws + waits + 1;
   VkPipelineStageFlags *stages = (VkPipelineStageFlags *)(ss + signals + 1);

   memcpy(ws, submit->pWaitSemaphores, waits * sizeof(*ws));
   memcpy(stages, submit->pWaitDstStageMask, waits * sizeof(*stages));
   memcpy(ss, submit->pSignalSemaphores, signals * sizeof(*ss));
   if (app_ti && app_ti->pWaitSemaphoreValues)
      memcpy(wv, app_ti->pWaitSemaphoreValues, MIN2(app_ti->waitSemaphoreValueCount, waits) * sizeof(*wv));
   if (app_ti && app_ti->pSignalSemaphoreValues)
      memcpy(sv, app_ti->pSignalSemaphoreValues, MIN2(app_ti->signalSemaphoreValueCount, signals) * sizeof(*sv));

   ws[waits] = device->hunt_chain;
   stages[waits] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   wv[waits] = device->hunt_chain_value;
   ss[signals] = device->hunt_chain;
   sv[signals] = ++device->hunt_chain_value;

   *ti = (VkTimelineSemaphoreSubmitInfo) {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .pNext = rest,
      .waitSemaphoreValueCount = waits + 1,
      .pWaitSemaphoreValues = wv,
      .signalSemaphoreValueCount = signals + 1,
      .pSignalSemaphoreValues = sv,
   };
   submit->pNext = ti;
   submit->waitSemaphoreCount = waits + 1;
   submit->pWaitSemaphores = ws;
   submit->pWaitDstStageMask = stages;
   submit->signalSemaphoreCount = signals + 1;
   submit->pSignalSemaphores = ss;
   return mem;
}
