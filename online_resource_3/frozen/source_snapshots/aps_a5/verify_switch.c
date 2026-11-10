/* Independent verifier: every proposed transition must agree with archived
   accepted trace. Updates a separately reconstructed state. */
#include <stdint.h>
#include <stddef.h>
int verify_proposals(uint8_t *state, const uint8_t *forbidden,
 const uint32_t *rindex, uint32_t *cindex, const uint32_t *pick1,
 const uint32_t *pick2, const uint8_t *stay, uint32_t proposals,
 uint32_t width, uint32_t origin, const uint32_t *events,
 uint32_t event_count) {
 uint32_t seen=0;
 for(uint32_t k=0;k<proposals;k++){
  uint32_t a=pick1[k],b=pick2[k];
  uint32_t ra=rindex[a],rb=rindex[b],ca=cindex[a],cb=cindex[b];
  size_t olda=(size_t)ra*width+ca,oldb=(size_t)rb*width+cb;
  size_t newa=(size_t)ra*width+cb,newb=(size_t)rb*width+ca;
  if(!state[olda] || !state[oldb])return 1;
  int permitted=(!stay[k] && ra!=rb && ca!=cb && !state[newa] && !state[newb] && !forbidden[newa] && !forbidden[newb]);
  int recorded=(seen<event_count && events[3*seen]==origin+k);
  if(permitted!=recorded)return 2;
  if(!permitted)continue;
  if(events[3*seen+1]!=a || events[3*seen+2]!=b)return 3;
  state[olda]=state[oldb]=0;state[newa]=state[newb]=1;
  cindex[a]=cb;cindex[b]=ca;seen++;
 }
 return seen==event_count?0:4;
}
