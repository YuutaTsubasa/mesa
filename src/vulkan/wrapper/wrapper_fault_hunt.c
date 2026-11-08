/*
 * WRAPPER_FAULT_HUNT=1: find which submission (and which pipelines in it)
 * makes the GPU fault.
 *
 * Some drivers -- the Mali proprietary blob on the Tab S12 Ultra is the one
 * this was written for -- report a GPU page fault only as a
 * "GROUP_ERROR_FATAL" line on the process's stdout/stderr and never return
 * VK_ERROR_DEVICE_LOST, so neither VK_EXT_device_fault nor vkd3d-proton's
 * breadcrumbs ever fire.  In this mode:
 *
 *  - stdout/stderr are routed through a pipe; a reader thread forwards every
 *    byte to where they went before and counts "GROUP_ERROR_FATAL" lines;
 *  - every vkQueueSubmit/vkQueueSubmit2 is serialised across all queues and
 *    followed by an empty submit with a private fence that is waited on
 *    (WRAPPER_FAULT_HUNT_MS, default 2000 ms), so a fault can be pinned on
 *    the batch that was just submitted;
 *  - command buffers remember the pipelines bound in them, and pipelines
 *    remember the 64-bit FNV-1a hash of each stage's SPIR-V as the
 *    application passed it (the same bytes vkd3d-proton dumps with
 *    VKD3D_SHADER_DUMP_PATH), so a report names the shaders involved.
 *
 * Reports go to stderr as "[FAULT_HUNT] ..." lines.  This is slow and meant
 * for debugging only; with the variable unset nothing here runs.
 */

#include "wrapper_private.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/hash_table.h"
#include "util/macros.h"
#include "util/simple_mtx.h"
#include "vk_util.h"

#define HUNT_MAX_STAGES 6
#define HUNT_MAX_REPORTS 40

struct hunt_pipeline {
   VkPipelineBindPoint bind_point;
   uint32_t stage_count;
   VkShaderStageFlagBits stages[HUNT_MAX_STAGES];
   uint64_t hashes[HUNT_MAX_STAGES];
};

static int hunt_enabled = -1;
static int hunt_wait = 1;   /* 0 none, 1 CPU wait, 2 GPU chain (QueueSubmit2) */
static uint64_t hunt_timeout_ns;
static simple_mtx_t hunt_mtx = SIMPLE_MTX_INITIALIZER;   /* tables */
static simple_mtx_t submit_mtx = SIMPLE_MTX_INITIALIZER; /* submissions */
static struct hash_table_u64 *module_hashes;    /* VkShaderModule -> uint64_t */
static struct hash_table_u64 *pipeline_info;    /* VkPipeline -> hunt_pipeline */
static volatile uint64_t fault_lines;
static uint64_t submit_seq;
static unsigned reports;
static int forward_fd = -1;

bool
wrapper_fault_hunt_enabled(void)
{
   if (hunt_enabled == -1) {
      const char *e = getenv("WRAPPER_FAULT_HUNT");
      hunt_enabled = e && atoi(e);
      const char *ms = getenv("WRAPPER_FAULT_HUNT_MS");
      hunt_timeout_ns = (uint64_t)(ms && atoi(ms) > 0 ? atoi(ms) : 2000) * 1000000ull;
      const char *w = getenv("WRAPPER_FAULT_HUNT_WAIT");
      hunt_wait = w ? atoi(w) : 1;
   }
   return hunt_enabled;
}

static uint64_t
hunt_fnv1a(const void *data, size_t size)
{
   const uint8_t *p = data;
   uint64_t h = 0xcbf29ce484222325ull;
   for (size_t i = 0; i < size; i++) {
      h ^= p[i];
      h *= 0x100000001b3ull;
   }
   return h;
}

/* Forward the redirected output and count the driver's fault lines.  A line
 * may be split across reads, so the tail of the previous read is kept. */
static void *
hunt_reader(void *arg)
{
   int fd = (int)(intptr_t)arg;
   static const char needle[] = "GROUP_ERROR_FATAL";
   char buf[4096 + sizeof(needle)];
   size_t carry = 0;

   for (;;) {
      ssize_t n = read(fd, buf + carry, sizeof(buf) - carry - 1);
      if (n < 0 && errno == EINTR)
         continue;
      if (n <= 0)
         break;
      for (ssize_t off = 0; off < n;) {
         ssize_t w = write(forward_fd, buf + carry + off, n - off);
         if (w <= 0)
            break;
         off += w;
      }
      size_t len = carry + n;
      buf[len] = '\0';
      for (char *p = buf; (p = strstr(p, needle)); p += sizeof(needle) - 1)
         __atomic_add_fetch(&fault_lines, 1, __ATOMIC_SEQ_CST);
      carry = MIN2(len, sizeof(needle) - 2);
      memmove(buf, buf + len - carry, carry);
   }
   return NULL;
}

void
wrapper_fault_hunt_init(struct wrapper_device *device)
{
   (void)device;
   if (!wrapper_fault_hunt_enabled())
      return;

   simple_mtx_lock(&hunt_mtx);
   if (!module_hashes) {
      module_hashes = _mesa_hash_table_u64_create(NULL);
      pipeline_info = _mesa_hash_table_u64_create(NULL);
   }
   bool start_reader = forward_fd < 0;
   if (start_reader)
      forward_fd = dup(2);
   simple_mtx_unlock(&hunt_mtx);

   if (!start_reader || forward_fd < 0)
      return;

   int fds[2];
   if (pipe(fds) != 0) {
      fprintf(stderr, "[FAULT_HUNT] pipe() failed, fault lines will not be counted\n");
      return;
   }
   fflush(stdout);
   fflush(stderr);
   dup2(fds[1], 1);
   dup2(fds[1], 2);
   close(fds[1]);

   pthread_t thread;
   pthread_create(&thread, NULL, hunt_reader, (void *)(intptr_t)fds[0]);
   pthread_detach(thread);
   fprintf(stderr, "[FAULT_HUNT] enabled: submissions are serialised, timeout %llu ms\n",
           (unsigned long long)(hunt_timeout_ns / 1000000ull));
}

void
wrapper_fault_hunt_module(VkShaderModule module, const VkShaderModuleCreateInfo *info)
{
   if (!wrapper_fault_hunt_enabled() || !module_hashes || !info)
      return;
   uint64_t *h = malloc(sizeof(*h));
   if (!h)
      return;
   *h = hunt_fnv1a(info->pCode, info->codeSize);
   simple_mtx_lock(&hunt_mtx);
   _mesa_hash_table_u64_insert(module_hashes, (uint64_t)module, h);
   simple_mtx_unlock(&hunt_mtx);
}

static uint64_t
hunt_stage_hash(const VkPipelineShaderStageCreateInfo *stage)
{
   const VkShaderModuleCreateInfo *inline_code =
      vk_find_struct_const(stage->pNext, SHADER_MODULE_CREATE_INFO);
   if (inline_code)
      return hunt_fnv1a(inline_code->pCode, inline_code->codeSize);
   if (stage->module) {
      uint64_t *h = _mesa_hash_table_u64_search(module_hashes, (uint64_t)stage->module);
      if (h)
         return *h;
   }
   return 0;
}

static void
hunt_add_stages(struct hunt_pipeline *hp, uint32_t count,
                const VkPipelineShaderStageCreateInfo *stages)
{
   for (uint32_t i = 0; i < count && hp->stage_count < HUNT_MAX_STAGES; i++) {
      hp->stages[hp->stage_count] = stages[i].stage;
      hp->hashes[hp->stage_count] = hunt_stage_hash(&stages[i]);
      hp->stage_count++;
   }
}

/* Linked graphics pipelines carry their stages in the libraries they link. */
static void
hunt_add_libraries(struct hunt_pipeline *hp, const void *pNext)
{
   const VkPipelineLibraryCreateInfoKHR *libs =
      vk_find_struct_const(pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   for (uint32_t i = 0; libs && i < libs->libraryCount; i++) {
      struct hunt_pipeline *lib =
         _mesa_hash_table_u64_search(pipeline_info, (uint64_t)libs->pLibraries[i]);
      for (uint32_t j = 0; lib && j < lib->stage_count && hp->stage_count < HUNT_MAX_STAGES; j++) {
         hp->stages[hp->stage_count] = lib->stages[j];
         hp->hashes[hp->stage_count] = lib->hashes[j];
         hp->stage_count++;
      }
   }
}

void
wrapper_fault_hunt_graphics(uint32_t count, const VkGraphicsPipelineCreateInfo *infos,
                            const VkPipeline *pipelines)
{
   if (!wrapper_fault_hunt_enabled() || !pipeline_info)
      return;
   simple_mtx_lock(&hunt_mtx);
   for (uint32_t i = 0; i < count; i++) {
      if (!pipelines[i])
         continue;
      struct hunt_pipeline *hp = calloc(1, sizeof(*hp));
      if (!hp)
         break;
      hp->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
      hunt_add_stages(hp, infos[i].stageCount, infos[i].pStages);
      hunt_add_libraries(hp, infos[i].pNext);
      _mesa_hash_table_u64_insert(pipeline_info, (uint64_t)pipelines[i], hp);
   }
   simple_mtx_unlock(&hunt_mtx);
}

void
wrapper_fault_hunt_compute(uint32_t count, const VkComputePipelineCreateInfo *infos,
                           const VkPipeline *pipelines)
{
   if (!wrapper_fault_hunt_enabled() || !pipeline_info)
      return;
   simple_mtx_lock(&hunt_mtx);
   for (uint32_t i = 0; i < count; i++) {
      if (!pipelines[i])
         continue;
      struct hunt_pipeline *hp = calloc(1, sizeof(*hp));
      if (!hp)
         break;
      hp->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
      hunt_add_stages(hp, 1, &infos[i].stage);
      _mesa_hash_table_u64_insert(pipeline_info, (uint64_t)pipelines[i], hp);
   }
   simple_mtx_unlock(&hunt_mtx);
}

void
wrapper_fault_hunt_begin(struct wrapper_command_buffer *wcb)
{
   wcb->hunt_count = 0;
   wcb->hunt_overflow = false;
   wcb->hunt_skip = false;
}

/* WRAPPER_FAULT_HUNT_SKIP: comma-separated stage hashes (as printed in the
 * reports).  Draws made while a graphics pipeline with any of those stages is
 * bound are dropped, to bisect which pipeline faults. */
#define HUNT_MAX_SKIP 64
static uint64_t skip_hashes[HUNT_MAX_SKIP];
static int skip_count = -1;

static bool
hunt_pipeline_skipped(VkPipeline pipeline)
{
   if (skip_count == -1) {
      skip_count = 0;
      const char *e = getenv("WRAPPER_FAULT_HUNT_SKIP");
      while (e && *e && skip_count < HUNT_MAX_SKIP) {
         char *end;
         uint64_t h = strtoull(e, &end, 16);
         if (end == e)
            break;
         skip_hashes[skip_count++] = h;
         e = *end ? end + 1 : end;
      }
      if (skip_count)
         fprintf(stderr, "[FAULT_HUNT] skipping draws of %d shader hash(es)\n", skip_count);
   }
   if (!skip_count)
      return false;

   bool skip = false;
   simple_mtx_lock(&hunt_mtx);
   struct hunt_pipeline *hp = _mesa_hash_table_u64_search(pipeline_info, (uint64_t)pipeline);
   for (uint32_t s = 0; hp && s < hp->stage_count && !skip; s++)
      for (int i = 0; i < skip_count && !skip; i++)
         skip = hp->hashes[s] == skip_hashes[i];
   simple_mtx_unlock(&hunt_mtx);
   return skip;
}

void
wrapper_fault_hunt_bind(struct wrapper_command_buffer *wcb, VkPipelineBindPoint bind_point,
                        VkPipeline pipeline)
{
   if (!wrapper_fault_hunt_enabled())
      return;
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS)
      wcb->hunt_skip = hunt_pipeline_skipped(pipeline);
   for (uint32_t i = 0; i < wcb->hunt_count; i++)
      if (wcb->hunt_pipes[i] == pipeline)
         return;
   if (wcb->hunt_count < ARRAY_SIZE(wcb->hunt_pipes))
      wcb->hunt_pipes[wcb->hunt_count++] = pipeline;
   else
      wcb->hunt_overflow = true;
}

static const char *
hunt_stage_name(VkShaderStageFlagBits s)
{
   switch (s) {
   case VK_SHADER_STAGE_VERTEX_BIT: return "vs";
   case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: return "hs";
   case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return "ds";
   case VK_SHADER_STAGE_GEOMETRY_BIT: return "gs";
   case VK_SHADER_STAGE_FRAGMENT_BIT: return "ps";
   case VK_SHADER_STAGE_COMPUTE_BIT: return "cs";
   default: return "??";
   }
}

static void
hunt_report_cb(uint32_t index, struct wrapper_command_buffer *wcb)
{
   fprintf(stderr, "[FAULT_HUNT]   cmdbuf[%u] %p: %u pipeline(s)%s\n", index,
           (void *)wcb, wcb->hunt_count, wcb->hunt_overflow ? " (list full, more not shown)" : "");
   simple_mtx_lock(&hunt_mtx);
   for (uint32_t i = 0; i < wcb->hunt_count; i++) {
      struct hunt_pipeline *hp =
         _mesa_hash_table_u64_search(pipeline_info, (uint64_t)wcb->hunt_pipes[i]);
      char line[512];
      int len = snprintf(line, sizeof(line), "[FAULT_HUNT]     pipeline 0x%llx %s:",
                         (unsigned long long)wcb->hunt_pipes[i],
                         !hp ? "unknown" :
                         hp->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? "compute" : "graphics");
      for (uint32_t s = 0; hp && s < hp->stage_count && len < (int)sizeof(line); s++)
         len += snprintf(line + len, sizeof(line) - len, " %s=%016llx",
                         hunt_stage_name(hp->stages[s]), (unsigned long long)hp->hashes[s]);
      fprintf(stderr, "%s\n", line);
   }
   simple_mtx_unlock(&hunt_mtx);
}

void
wrapper_fault_hunt_submit_begin(void)
{
   if (wrapper_fault_hunt_enabled())
      simple_mtx_lock(&submit_mtx);
}

/* Called with the submit lock held, right after the driver accepted the batch.
 * cmdbufs/count are the application's (wrapper) command buffers of the batch. */
bool
wrapper_fault_hunt_gpu_chain(void)
{
   return wrapper_fault_hunt_enabled() && hunt_wait == 2;
}

/* WRAPPER_FAULT_HUNT_WAIT=2: make this batch wait for the previous batch (on
 * any queue) and signal its own completion, so the GPU runs batches strictly
 * one after another without the CPU stalling.  Called with the submit lock
 * held.  Returns memory to free after the submit, or NULL. */
void *
wrapper_fault_hunt_chain2(struct wrapper_device *device, VkSubmitInfo2 *submit)
{
   if (!wrapper_fault_hunt_gpu_chain())
      return NULL;

   if (!device->hunt_chain) {
      VkSemaphoreTypeCreateInfo type = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      };
      VkSemaphoreCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type,
      };
      if (device->dispatch_table.CreateSemaphore(device->dispatch_handle, &sci, NULL,
                                                 &device->hunt_chain) != VK_SUCCESS)
         return NULL;
      device->hunt_chain_value = 0;
   }

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

void
wrapper_fault_hunt_submit_end(struct wrapper_queue *queue, VkResult submit_result,
                              struct wrapper_command_buffer *const *cmdbufs, uint32_t count,
                              bool chained)
{
   if (!wrapper_fault_hunt_enabled())
      return;

   struct wrapper_device *device = queue->device;
   uint64_t before = __atomic_load_n(&fault_lines, __ATOMIC_SEQ_CST);
   uint64_t seq = ++submit_seq;
   bool timed_out = false;

   if (submit_result == VK_SUCCESS && (hunt_wait == 1 || (hunt_wait == 2 && !chained))) {
      VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
      VkFence fence = VK_NULL_HANDLE;
      if (device->dispatch_table.CreateFence(device->dispatch_handle, &fci, NULL, &fence) == VK_SUCCESS) {
         if (device->dispatch_table.QueueSubmit(queue->dispatch_handle, 0, NULL, fence) == VK_SUCCESS)
            timed_out = device->dispatch_table.WaitForFences(device->dispatch_handle, 1, &fence,
                                                             VK_TRUE, hunt_timeout_ns) == VK_TIMEOUT;
         device->dispatch_table.DestroyFence(device->dispatch_handle, fence, NULL);
      }
      /* The driver's event thread prints the fault a little after the
       * group dies; give it a moment before deciding the batch was clean. */
      for (int i = 0; i < 10 && __atomic_load_n(&fault_lines, __ATOMIC_SEQ_CST) == before; i++)
         usleep(2000);
   }

   uint64_t after = __atomic_load_n(&fault_lines, __ATOMIC_SEQ_CST);
   if ((after != before || timed_out || submit_result != VK_SUCCESS) && reports < HUNT_MAX_REPORTS) {
      reports++;
      fprintf(stderr, "[FAULT_HUNT] submit #%llu on queue %p: %llu fault line(s)%s, result %d, %u cmdbuf(s)\n",
              (unsigned long long)seq, (void *)queue, (unsigned long long)(after - before),
              timed_out ? ", fence TIMED OUT" : "", submit_result, count);
      for (uint32_t i = 0; i < count; i++)
         hunt_report_cb(i, cmdbufs[i]);
   }
   simple_mtx_unlock(&submit_mtx);
}
