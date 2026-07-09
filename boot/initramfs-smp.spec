# gen_init_cpio spec (CONFIG_INITRAMFS_SOURCE) for the T5.1c SMP dual-hart fuzz kernel
# (firmware/Image.smp). Identical to boot/initramfs.spec except it packs boot/agent-smp.c's
# build (build/agent-smp) as /init instead of boot/agent.c's (build/agent) — see
# boot/agent-smp.c's header comment for why this needs a genuinely separate kernel image rather
# than modifying the shared boot/agent.c + firmware/Image the single-hart path and `smp-boot`
# already use.
dir /dev 0755 0 0
nod /dev/console 0600 0 0 c 5 1
nod /dev/null 0666 0 0 c 1 3
dir /proc 0755 0 0
dir /sys 0755 0 0
dir /sys/kernel 0755 0 0
dir /sys/kernel/debug 0755 0 0
file /init /home/forrest/fuzzsoft/build/agent-smp 0755 0 0
