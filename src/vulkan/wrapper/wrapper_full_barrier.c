/*
 * WRAPPER_FULL_BARRIERS: test whether the GPU faults come from work inside
 * one command buffer running out of order.
 *
 *  1: after every pipeline barrier / event wait the application records,
 *     add a global ALL_COMMANDS -> ALL_COMMANDS, MEMORY_WRITE ->
 *     MEMORY_READ|MEMORY_WRITE barrier.  Fixes the case where the driver
 *     mishandles particular stage/access masks (e.g. TRANSFER -> COMPUTE
 *     after vkCmdUpdateBuffer, which vkd3d-proton's query gather relies on).
 *  2: as 1, and also add that barrier after every dispatch, buffer copy /
 *     update / fill, query-pool copy and dynamic rendering pass.  Covers
 *     dependencies the application forgot to synchronise as well.
 *
 * Mali-G1-Ultra (Sonic Racing CrossWorlds on vkd3d-proton): with
 * WRAPPER_CHAIN_SUBMITS=1 the faults drop from 100+ to a few, but the
 * remaining ones still kill the game.  Barriers are never added inside a
 * dynamic rendering pass.
 */

#include "wrapper_private.h"
#include "wrapper_log.h"

#include <stdlib.h>

#include "wrapper_entrypoints.h"

static int full_barrier_mode = -1;

int
wrapper_full_barrier_mode(void)
{
   if (full_barrier_mode == -1) {
      const char *e = getenv("WRAPPER_FULL_BARRIERS");
      full_barrier_mode = e ? atoi(e) : 0;
      if (full_barrier_mode)
         WRAPPER_LOG(info, "WRAPPER_FULL_BARRIERS=%d", full_barrier_mode);
   }
   return full_barrier_mode;
}

void
wrapper_full_barrier(struct wrapper_command_buffer *wcb, int min_mode)
{
   if (wrapper_full_barrier_mode() < min_mode || wcb->dynamic_rendering_active)
      return;
   VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   wcb->device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
      1, &barrier, 0, NULL, 0, NULL);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdDispatch(wcb->dispatch_handle, x, y, z);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t bx, uint32_t by,
                        uint32_t bz, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdDispatchBase(wcb->dispatch_handle, bx, by, bz, x, y, z);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                            VkDeviceSize offset)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdDispatchIndirect(wcb->dispatch_handle, buffer, offset);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
                        VkDeviceSize offset, VkDeviceSize size, const void *data)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdUpdateBuffer(wcb->dispatch_handle, buffer, offset, size,
                                               data);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
                      VkDeviceSize offset, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdFillBuffer(wcb->dispatch_handle, buffer, offset, size,
                                             data);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst,
                      uint32_t count, const VkBufferCopy *regions)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, src, dst, count,
                                             regions);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdCopyBuffer2(wcb->dispatch_handle, info);
   wrapper_full_barrier(wcb, 2);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool pool,
                                uint32_t first, uint32_t count, VkBuffer dst,
                                VkDeviceSize offset, VkDeviceSize stride,
                                VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdCopyQueryPoolResults(wcb->dispatch_handle, pool, first,
                                                       count, dst, offset, stride, flags);
   wrapper_full_barrier(wcb, 2);
}
