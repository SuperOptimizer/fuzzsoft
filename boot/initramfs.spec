# gen_init_cpio spec (CONFIG_INITRAMFS_SOURCE). Creates /dev/console so the kernel can wire
# init's stdin/stdout/stderr to the console, plus our tiny /init. No root/mknod needed.
dir /dev 0755 0 0
nod /dev/console 0600 0 0 c 5 1
nod /dev/null 0666 0 0 c 1 3
# Mountpoints for the fault-injection agent preamble (boot/agent.c's
# arm_fault_injection_knobs()): proc/sysfs/debugfs need their target directories to already
# exist before mount() will succeed. Harmless on kernels that don't use them (stock/slubdebug) —
# just empty directories that are never mounted onto.
dir /proc 0755 0 0
dir /sys 0755 0 0
dir /sys/kernel 0755 0 0
dir /sys/kernel/debug 0755 0 0
file /init /home/forrest/fuzzsoft/build/agent 0755 0 0
