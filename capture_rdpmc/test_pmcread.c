#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct pmc_req {
    uint32_t cpu;
    uint32_t ecx;
    uint64_t value;
};
#define PMCR_IOC_MAGIC   'p'
// #define "ioctl name" __IOX("magic number","command number","argument type")
// _IOWR: ioctl with both write and read parameters
#define PMCR_IOC_READ _IOWR(PMCR_IOC_MAGIC, 0x01, struct pmc_req)

int main(int argc, char **argv)
{
    int fd = open("/dev/pmcread", O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    struct pmc_req r = {0};
    r.cpu = 0;

    // Example: fixed counter 1 (unhalted core cycles)
    // RDPMC selector: bit30=1 => fixed, lower bits = index
    r.ecx = (1u<<30) | 1u;

    // Include the header file <sys/ioctl.h>.Now we need to call the ioctl command from a user application.
    // long ioctl( "file descriptor","ioctl command","Arguments");

    if (ioctl(fd, PMCR_IOC_READ, &r) != 0) { perror("ioctl"); return 1; }
    printf("FIXED1 on CPU%u = %llu\n", r.cpu, (unsigned long long)r.value);

    close(fd);
    return 0;
}
