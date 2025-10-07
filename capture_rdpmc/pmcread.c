#include <linux/module.h>      // Core module support
#include <linux/miscdevice.h>  // Misc device framework
#include <linux/fs.h>          // File operations
#include <linux/uaccess.h>     // copy_from_user/copy_to_user
#include <linux/smp.h>         // Multi-processor support
#include <linux/cpu.h>
#include <linux/sched.h>
#include <linux/version.h>
#include <asm/msr.h>   // native_read_pmc(), rdpmc(int)

#define PMCR_IOC_MAGIC   'p'
struct pmc_req {
    __u32 cpu;      // target logical CPU
    __u32 ecx;      // RDPMC selector (bit 30 => fixed; else programmable)
    __u64 value;    // out
};

// #define "ioctl name" __IOX("magic number","command number","argument type")
// _IOWR: ioctl with both write and read parameters
#define PMCR_IOC_READ _IOWR(PMCR_IOC_MAGIC, 0x01, struct pmc_req)

static void pmc_read_on_cpu(void *info)
{
    struct pmc_req *r = info;
    // Runs on target CPU thanks to smp_call_function_single()
    // native_read_pmc() from arch/x86/include/asm/msr.h
    // this function supports trace for RDPMC
    r->value = native_read_pmc(r->ecx); // emits tracepoint if enabled
}
// This handles requests from userspace
static long pmcr_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct pmc_req req;
    int ret = 0;

    if (cmd != PMCR_IOC_READ)
        return -ENOTTY;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    if (req.cpu >= nr_cpu_ids || !cpu_online(req.cpu))
        return -ENXIO;

    // Run the read on that CPU
    /* * smp_call_function_single - Run a function on a specific CPU
    * @cpu: The CPU to run the function on.
    * @func: The function to run. This must be fast and non-blocking.
    * @info: An arbitrary pointer to pass to the function.
    * @wait: If true, wait until function has completed on other CPUs.
    */
    ret = smp_call_function_single(req.cpu, pmc_read_on_cpu, &req, 1);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &req, sizeof(req)))
        return -EFAULT;

    return 0;
}

static const struct file_operations pmcr_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = pmcr_ioctl,
#ifdef CONFIG_COMPAT
    .compat_ioctl   = pmcr_ioctl,
#endif
};

static struct miscdevice pmcr_dev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = "pmcread",
    .fops  = &pmcr_fops,
    .mode  = 0600, // root-only; relax if you must (consider CAP_SYS_ADMIN)
};

static int __init pmcr_init(void)
{
    return misc_register(&pmcr_dev);
}
static void __exit pmcr_exit(void)
{
    misc_deregister(&pmcr_dev);
}

module_init(pmcr_init);
module_exit(pmcr_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("mk");
MODULE_DESCRIPTION("Expose kernel-side RDPMC via ioctl");
