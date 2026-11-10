#include <stdint.h>
#include <stddef.h>
/* Independently written verifier: enumerate all proposals and reproduce
   accepted records; no invocation of the generation implementation. */
size_t verify_proposals(uint8_t *state,const uint8_t *permitted,
 const uint32_t *row,uint32_t *column,const uint32_t *left,
 const uint32_t *right,const uint8_t *stay,size_t count,
 uint32_t width,uint32_t offset,uint32_t *records) {
 size_t n=0;
 for(size_t t=0;t<count;t++) {
  uint32_t u=left[t],v=right[t];
  if(stay[t]!=0 || row[u]==row[v] || column[u]==column[v]) continue;
  size_t old_u=(size_t)row[u]*width+column[u];
  size_t old_v=(size_t)row[v]*width+column[v];
  size_t new_u=(size_t)row[u]*width+column[v];
  size_t new_v=(size_t)row[v]*width+column[u];
  if(!permitted[new_u] || !permitted[new_v]) continue;
  if(state[new_u]!=0 || state[new_v]!=0) continue;
  if(state[old_u]!=1 || state[old_v]!=1) return SIZE_MAX;
  state[old_u]=0; state[old_v]=0;
  state[new_u]=1; state[new_v]=1;
  uint32_t save=column[u];column[u]=column[v];column[v]=save;
  records[n*3]=(uint32_t)t+offset;records[n*3+1]=u;records[n*3+2]=v;n++;
 }
 return n;
}
