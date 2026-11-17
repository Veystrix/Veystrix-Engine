#include "qsa_full.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <limits>
namespace mx = mlx::core;
using namespace veystrix::engine;
static void sv(const char* n, const mx::array& a){ mx::eval(a); mx::save(std::string("/tmp/dbg_c_")+n+".npy", a); }
int main(int argc, char** argv){
  std::string dir = argc>1?argv[1]:"cpp/engine/tests/vectors_qsa_full/case1_select";
  auto L=[&](const char* n){ return mx::load(dir+"/"+n+".npy"); };
  mx::array x=L("x"), pw=L("proj_weight"), qw=L("q_weight"), kw=L("k_weight");
  mx::array pk=L("past_index_keys"), pp=L("past_index_positions"), pos=L("position_ids");
  QSAFullConfig c; c.base.n_heads=2; c.base.kv_heads=1; c.base.head_dim=128;
  c.base.compress_ratio=4; c.base.block_topk=4; c.base.eps=1e-6f;
  c.rope.dim=32; c.rope.base=10000000.0f; c.rope.mrope_section={11,11,10};
  c.rope.style="interleaved"; c.rope.attention_scaling=1.0f;
  int B=x.shape()[0], S=x.shape()[1], Hq=2, Hk=1, D=128, ratio=4, topk=4, past_len=24;

  mx::array qk=mx::matmul(x, mx::transpose(pw));
  qk=mx::reshape(qk,{B,S,Hq+Hk,D});
  mx::array query=mx::slice(qk,{0,0,0,0},{B,S,Hq,D},{1,1,1,1});
  mx::array raw=mx::squeeze(mx::slice(qk,{0,0,Hq,0},{B,S,Hq+Hk,D},{1,1,1,1}),2);
  query=mx::transpose(qsa_rms_norm(query,qw,c.base.eps),{0,2,1,3});
  sv("query_pre",query);

  QSACache cache; cache.offset=past_len; cache.update_indexer(pk,pp);
  cache.update_indexer(raw,pos);
  mx::array full=*cache.index_keys, full_pos=*cache.index_position_ids;
  int key_len=full.shape()[1], blocks=key_len/ratio, clen=blocks*ratio;
  mx::array pooled=mx::slice(full,{0,0,0},{B,clen,D},{1,1,1});
  pooled=mx::reshape(pooled,{B,blocks,ratio,D});
  pooled=mx::astype(mx::mean(mx::astype(pooled,mx::float32),2),full.dtype());
  pooled=qsa_rms_norm(pooled,kw,c.base.eps);
  pooled=mx::expand_dims(pooled,1);
  sv("pooled_pre",pooled);

  query=qsa_apply_rope(query,pos,c.rope); sv("query_roped",query);
  mx::array bs=mx::astype(mx::arange(blocks),mx::int32)*ratio;
  mx::array bp=mx::take(full_pos,bs,-1);
  pooled=qsa_apply_rope(pooled,bp,c.rope); sv("pooled_roped",pooled);

  mx::array scores=mx::matmul(query,mx::transpose(pooled,{0,1,3,2})); sv("scores_matmul",scores);
  mx::array zero(0.0f,mx::float32);
  scores=mx::sum(mx::maximum(mx::astype(scores,mx::float32),zero),1); sv("scores_sum",scores);
  double sd=std::sqrt(128.0); mx::array scale((float)sd,mx::float32);
  scores=scores/scale; sv("scores_scaled",scores);

  mx::array qe=mx::astype(mx::arange(past_len+1,past_len+S+1),mx::int32);
  mx::array cc=qe/ratio;
  mx::array valid=mx::less(mx::expand_dims(mx::expand_dims(mx::astype(mx::arange(blocks),mx::int32),0),0),
                           mx::expand_dims(cc,{0,2}));
  mx::array neginf(-std::numeric_limits<float>::infinity(),mx::float32);
  scores=mx::where(valid,scores,neginf); sv("scores_masked",scores);
  mx::array sel=mx::argpartition(scores,-topk,-1);
  sel=mx::slice(sel,{0,0,blocks-topk},{B,S,blocks},{1,1,1});
  sv("selected_blocks",sel);

  mx::array token_indices=mx::astype(mx::arange(key_len),mx::int32);
  mx::array token_blocks=token_indices/ratio;
  mx::array tb=mx::expand_dims(token_blocks,{0,1,2});
  mx::array sb=mx::expand_dims(sel,-1);
  sv("tb",tb); sv("sb",sb); sv("eq",mx::equal(tb,sb));
  mx::array st=mx::any(mx::equal(tb,sb),2); sv("sel_any",st);
  mx::array ccarr=qe/ratio;
  mx::array tail_starts=ccarr*ratio;
  mx::array ti=mx::expand_dims(token_indices,{0,1});
  mx::array ts=mx::expand_dims(tail_starts,{0,2});
  mx::array qex=mx::expand_dims(qe,{0,2});
  mx::array tail=mx::logical_and(mx::greater_equal(ti,ts),mx::less(ti,qex));
  mx::array causal=mx::less(ti,qex);
  mx::array topkv(topk,ccarr.dtype());
  mx::array us=mx::expand_dims(mx::greater(ccarr,topkv),{0,2});
  st=mx::where(us,mx::logical_or(st,tail),causal);
  sv("sel_final",st);
  return 0;
}
