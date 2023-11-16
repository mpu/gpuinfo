#pragma once

void *allocWorkBuffer(size_t);
uint launchSumKernel(uint *, size_t, void *);
void launchFillKernel(uint *, size_t);
