#include <stdint.h>
#include <stddef.h>
/* Each proposal selects two indexed edges (rows fixed; columns evolve).
   Mask and allowed are C-order uint8. Return accepted trace count.
   Symmetric two-edge exchange; all rejections and lazy draws stay put. */
size_t a5_switch(uint8_t *mask, const uint8_t *allowed, const uint32_t *rows,
                 uint32_t *cols, const uint32_t *a, const uint32_t *b,
                 const uint8_t *lazy, size_t n, uint32_t width,
                 uint32_t offset, uint32_t *trace) {
    size_t accepted=0;
    for(size_t k=0;k<n;k++) {
        if(lazy[k]) continue;
        uint32_t i=a[k], j=b[k], r=rows[i], s=rows[j], c=cols[i], d=cols[j];
        if(r==s || c==d) continue;
        size_t rd=(size_t)r*width+d, sc=(size_t)s*width+c;
        if(mask[rd] || mask[sc] || !allowed[rd] || !allowed[sc]) continue;
        mask[(size_t)r*width+c]=0; mask[(size_t)s*width+d]=0;
        mask[rd]=1;mask[sc]=1;cols[i]=d;cols[j]=c;
        trace[accepted*3]=offset+(uint32_t)k;
        trace[accepted*3+1]=i;trace[accepted*3+2]=j;accepted++;
    }
    return accepted;
}
