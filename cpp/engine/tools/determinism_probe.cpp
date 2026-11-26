// determinism_probe -- isolate WHICH MLX op is nondeterministic in the
// real_checkpoint_generate lm_head path.
//
// This tool uses SYNTHETIC inputs with the EXACT shapes/dtypes of the ops used
// by the real generator, so it runs in seconds instead of loading the 105 GB
// checkpoint. For every op it evaluates the SAME inputs 5 times in one process
// and prints a SHA-256 (float32-LE, the generator's convention) plus the number
// of differing elements and max abs difference vs. run 0.
//
// Build (from repo root):
//   MLXROOT=.venv/lib/python3.12/site-packages/mlx
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp \
//     cpp/engine/src/gdn_conv.cpp cpp/engine/src/gdn_layer.cpp \
//     cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
//     cpp/determinism_probe_dummy.cpp 2>/dev/null || true
//   # real build (engine.cpp needed for gated_delta_step_kernel fallback):
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp \
//     cpp/engine/tools/determinism_probe.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/det_probe
//
// Usage: /tmp/det_probe [qmm|dqmm|gdn|sdpa|all]
#include <mlx/mlx.h>
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

// ------------------------------- SHA-256 ---------------------------------- //
namespace {
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                   0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  static uint32_t rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
  static void compress(uint32_t h[8],const uint8_t b[64]){
    static const uint32_t K[64]={
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for(int i=0;i<16;++i) w[i]=(uint32_t(b[4*i])<<24)|(uint32_t(b[4*i+1])<<16)|(uint32_t(b[4*i+2])<<8)|uint32_t(b[4*i+3]);
    for(int i=16;i<64;++i){uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
    uint32_t a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for(int i=0;i<64;++i){uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25),ch=(e&f)^((~e)&g),t1=hh+S1+ch+K[i]+w[i];uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22),maj=(a&bb)^(a&c)^(bb&c),t2=S0+maj;hh=g;g=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;}
    h[0]+=a;h[1]+=bb;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
  }
};
std::string sha256_hex(const void* data,size_t len){
  Sha256 s;size_t total=((len+9+63)/64)*64;std::vector<uint8_t> buf(total,0);
  const uint8_t* in=static_cast<const uint8_t*>(data);for(size_t i=0;i<len;++i)buf[i]=in[i];buf[len]=0x80;
  uint64_t bits=uint64_t(len)*8;for(int i=0;i<8;++i)buf[total-1-i]=uint8_t(bits>>(8*i));
  for(size_t i=0;i<total;i+=64)Sha256::compress(s.h,buf.data()+i);
  static const char* HD="0123456789abcdef";std::string out(64,'0');
  for(int i=0;i<8;++i)for(int j=0;j<4;++j){out[i*8+j*2]=HD[(s.h[i]>>(28-8*j))&0xf];out[i*8+j*2+1]=HD[(s.h[i]>>(24-8*j))&0xf];}
  return out;
}

// Hash the raw float32-LE bytes of an array (casting through float32).
mx::array as_f32(const mx::array& a){return mx::astype(a,mx::float32);}
std::vector<float> to_host(const mx::array& a){
  mx::array f=as_f32(a);mx::eval(f);
  const float* p=f.data<float>();return std::vector<float>(p,p+f.size());
}
std::string hash_vec(const std::vector<float>& v){return sha256_hex(v.data(),v.size()*sizeof(float));}

// Build a fresh bfloat16/float32 array from host float data each call so no
// array/donation state is shared between repeated evaluations.
mx::array make_array(const std::vector<float>& v,mx::Shape shape,mx::Dtype dt){
  if(dt==mx::float32) return mx::array(v.data(),shape,mx::float32);
  std::vector<uint16_t> u(v.size());
  for(size_t i=0;i<v.size();++i){
    float x=v[i];uint32_t bits;std::memcpy(&bits,&x,4);
    // truncate fp32->bf16 (round-to-nearest-even)
    uint32_t lsb=bits&0x7fffu;uint32_t rounding_bias=0x7fffu+lsb+((bits>>16)&1u);
    u[i]=uint16_t((bits+rounding_bias)>>16);
  }
  return mx::array(u.data(),shape,mx::bfloat16);
}

void report(const char* tag,size_t run,const std::string& digest,
            const std::vector<float>& cur,const std::vector<float>& ref){
  if(run==0){std::printf("[%s] run0 size=%zu sha256=%s\n",tag,cur.size(),digest.c_str());return;}
  size_t diff=0;float maxd=0.0f;
  for(size_t i=0;i<cur.size()&&i<ref.size();++i){
    float d=std::fabs(cur[i]-ref[i]);if(d!=0.0f){++diff;maxd=std::max(maxd,d);}
  }
  std::printf("[%s] run%zu sha256=%s  diff=%zu/%zu maxabs=%.9g %s\n",
              tag,run,digest.c_str(),diff,cur.size(),maxd,
              digest==hash_vec(ref)?"IDENTICAL":"VARIES");
}

std::vector<float> rand_floats(size_t n,uint64_t seed,float scale=1.0f){
  std::mt19937_64 rng(seed);std::uniform_real_distribution<float> d(-scale,scale);
  std::vector<float> v(n);for(auto& x:v)x=d(rng);return v;
}
std::vector<uint16_t> rand_bf16_bits(size_t n,uint64_t seed){
  std::mt19937_64 rng(seed);std::vector<uint16_t> v(n);
  for(auto& x:v)x=uint16_t(0x3c00u|(rng()&0x03ffu)); // ~1.0 .. 1.998
  return v;
}

// --------------------------- (a) lm_head qmm ------------------------------ //
void probe_qmm(bool dense){
  const int N=248320,K=2560,GS=32,BITS=4;
  const int WPACK=K/8, SCALE_N=K/GS;
  const char* tag=dense?"dqmm":"qmm";
  std::printf("\n=== %s : x[1,1,%d] @ w[%d,%d] (gs=%d,bits=%d) ===\n",
              tag,K,N,WPACK,GS,BITS);

  // Packed weight + scales/biases (bf16), generated once on host.
  std::vector<uint32_t> wbits((size_t)N*WPACK);
  {std::mt19937_64 rng(12345);for(auto& x:wbits)x=uint32_t(rng());}
  auto sbits=rand_bf16_bits((size_t)N*SCALE_N,777);
  auto bbits=rand_bf16_bits((size_t)N*SCALE_N,999);

  for(mx::Dtype xdt:{mx::float32,mx::bfloat16}){
    const char* subt=xdt==mx::float32?"f32":"bf16";
    char tagdt[32];std::snprintf(tagdt,sizeof(tagdt),"%s-%s",tag,subt);
    std::printf("-- input dtype=%s --\n",xdt==mx::float32?"float32":"bfloat16");
    std::vector<float> xf=rand_floats(K,20261117,0.02f);
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array x=make_array(xf,mx::Shape{1,1,K},xdt);
      mx::array w(wbits.data(),mx::Shape{N,WPACK},mx::uint32);
      mx::array s(sbits.data(),mx::Shape{N,SCALE_N},mx::bfloat16);
      mx::array b(bbits.data(),mx::Shape{N,SCALE_N},mx::bfloat16);
      std::optional<mx::array> out;
      if(dense){
        mx::array wd=mx::dequantize(w,s,b,GS,BITS,"affine");
        mx::eval(wd);
        out=mx::matmul(x,mx::transpose(wd));
      }else{
        out=mx::quantized_matmul(x,w,s,b,/*transpose=*/true,GS,BITS,"affine");
      }
      std::vector<float> cur=to_host(*out);
      if(r==0)ref=cur;
      report(tagdt,r,hash_vec(cur),cur,ref);
    }
  }
}

// --------------------------- (b) GDN kernel -------------------------------- //
void probe_gdn(){
  const int B=1,T=1,HK=16,HV=48,DK=128,DV=128;
  std::printf("\n=== gdn fused kernel: q/k[1,1,%d,%d] v[1,1,%d,%d] state[1,%d,%d,%d] ===\n",
              HK,DK,HV,DV,HV,DV,DK);
  size_t nq=(size_t)B*T*HK*DK,nv=(size_t)B*T*HV*DV,ng=(size_t)B*T*HV,
         ns=(size_t)B*HV*DV*DK;
  auto qh=rand_floats(nq,1,1.0f),kh=rand_floats(nq,2,1.0f),vh=rand_floats(nv,3,1.0f);
  auto gh=rand_floats(ng,4,1.0f),bh=rand_floats(ng,5,1.0f),sh=rand_floats(ns,6,0.1f);
  std::vector<float> refy,refs;
  for(int r=0;r<5;++r){
    mx::array q(qh.data(),mx::Shape{B,T,HK,DK},mx::float32);
    mx::array k(kh.data(),mx::Shape{B,T,HK,DK},mx::float32);
    mx::array v(vh.data(),mx::Shape{B,T,HV,DV},mx::float32);
    mx::array g(gh.data(),mx::Shape{B,T,HV},mx::float32);
    mx::array beta(bh.data(),mx::Shape{B,T,HV},mx::float32);
    mx::array state(sh.data(),mx::Shape{B,HV,DV,DK},mx::float32);
    DeltaStepOutput o=gated_delta_step_kernel(q,k,v,g,beta,state);
    std::vector<float> y=to_host(o.y),s=to_host(o.state);
    std::vector<float> cat;cat.reserve(y.size()+s.size());
    cat.insert(cat.end(),y.begin(),y.end());cat.insert(cat.end(),s.begin(),s.end());
    if(r==0){refy=y;refs=s;std::printf("[gdn] run0 sha256=%s (y=%zu state=%zu)\n",hash_vec(cat).c_str(),y.size(),s.size());}
    else{
      size_t diff=0;float maxd=0;for(size_t i=0;i<y.size();++i){float d=std::fabs(y[i]-refy[i]);if(d!=0){++diff;maxd=std::max(maxd,d);}}
      for(size_t i=0;i<s.size();++i){float d=std::fabs(s[i]-refs[i]);if(d!=0){++diff;maxd=std::max(maxd,d);}}
      std::printf("[gdn] run%d sha256=%s diff=%zu maxabs=%.9g %s\n",r,hash_vec(cat).c_str(),diff,maxd,diff?"VARIES":"IDENTICAL");
    }
  }
}

// --------------- (e) MoE gather_qmm (host-backed slot buffers) -------------- //
void probe_gqmm(){
  const int SLOTS=16,K=2560,GS=32,BITS=4,TOK=1,TOPK=10;
  const int WPACK=K/8,SN=K/GS;
  std::printf("\n=== MoE gather_qmm: x[1,1,1,%d] w[%d,%d,%d] rhs[1,%d] ===\n",
              K,SLOTS,K,WPACK,TOPK);
  std::vector<uint32_t> wbits((size_t)SLOTS*K*WPACK);
  {std::mt19937_64 rng(555);for(auto& x:wbits)x=uint32_t(rng());}
  auto sbits=rand_bf16_bits((size_t)SLOTS*K*SN,556);
  auto bbits=rand_bf16_bits((size_t)SLOTS*K*SN,557);
  auto xf=rand_floats(TOK*K,558,0.02f);
  int32_t ri[TOPK];for(int i=0;i<TOPK;++i)ri[i]=(i*37)%SLOTS;
  std::vector<float> ref;
  for(int r=0;r<5;++r){
    mx::array x(xf.data(),mx::Shape{TOK,1,1,K},mx::bfloat16);
    // Host-backed arrays exactly like StreamingMoE::wrap (no-op deleter).
    auto keepw=wbits;auto keeps=sbits;auto keepb=bbits;
    mx::array w(wbits.data(),mx::Shape{SLOTS,K,WPACK},mx::uint32,[&](void*){});
    mx::array s(sbits.data(),mx::Shape{SLOTS,K,SN},mx::bfloat16,[&](void*){});
    mx::array b(bbits.data(),mx::Shape{SLOTS,K,SN},mx::bfloat16,[&](void*){});
    mx::array rhs(ri,mx::Shape{TOK,TOPK},mx::int32);
    mx::array out=mx::gather_qmm(x,w,s,b,std::nullopt,rhs,true,GS,BITS,"affine",false);
    std::vector<float> cur=to_host(out);
    if(r==0)ref=cur;
    report("gqmm",r,hash_vec(cur),cur,ref);
  }
}

// ------- (g) host-buffer lifetime: is mx::eval synchronous? ----------------- //
void probe_race(){
  const int SLOTS=16,K=2560,GS=32,BITS=4,TOK=1,TOPK=10,SN=K/GS,WPACK=K/8;
  std::printf("\n=== host-buffer/race: gather_qmm + immediate host mutation ===\n");
  std::vector<uint32_t> wbits((size_t)SLOTS*K*WPACK);
  auto sbits=rand_bf16_bits((size_t)SLOTS*K*SN,661);
  auto bbits=rand_bf16_bits((size_t)SLOTS*K*SN,662);
  auto xf=rand_floats(TOK*K,663,0.02f);
  int32_t ri[TOPK];for(int i=0;i<TOPK;++i)ri[i]=(i*37)%SLOTS;
  // Precompute two reference results (all-A weights vs all-B weights).
  auto run_once=[&](uint32_t fill){
    std::fill(wbits.begin(),wbits.end(),fill);
    mx::array x(xf.data(),mx::Shape{TOK,1,1,K},mx::bfloat16);
    mx::array w(wbits.data(),mx::Shape{SLOTS,K,WPACK},mx::uint32,[](void*){});
    mx::array s(sbits.data(),mx::Shape{SLOTS,K,SN},mx::bfloat16,[](void*){});
    mx::array b(bbits.data(),mx::Shape{SLOTS,K,SN},mx::bfloat16,[](void*){});
    mx::array rhs(ri,mx::Shape{TOK,TOPK},mx::int32);
    mx::array out=mx::gather_qmm(x,w,s,b,std::nullopt,rhs,true,GS,BITS,"affine",false);
    return to_host(out);
  };
  auto refA=run_once(0x11111111u), refB=run_once(0x22222222u);
  std::printf(" refA=%s\n refB=%s\n",hash_vec(refA).c_str(),hash_vec(refB).c_str());
  for(int r=0;r<5;++r){
    uint32_t fill=(r%2==0)?0x11111111u:0x22222222u;
    std::fill(wbits.begin(),wbits.end(),fill);
    mx::array x(xf.data(),mx::Shape{TOK,1,1,K},mx::bfloat16);
    mx::array w(wbits.data(),mx::Shape{SLOTS,K,WPACK},mx::uint32,[](void*){});
    mx::array s(sbits.data(),mx::Shape{SLOTS,K,SN},mx::bfloat16,[](void*){});
    mx::array b(bbits.data(),mx::Shape{SLOTS,K,SN},mx::bfloat16,[](void*){});
    mx::array rhs(ri,mx::Shape{TOK,TOPK},mx::int32);
    mx::array out=mx::gather_qmm(x,w,s,b,std::nullopt,rhs,true,GS,BITS,"affine",false);
    std::vector<float> cur=to_host(out);
    // Mutate host buffer immediately after eval (mimics next step's pread).
    std::fill(wbits.begin(),wbits.end(),~fill);
    bool matchesA=hash_vec(cur)==hash_vec(refA), matchesB=hash_vec(cur)==hash_vec(refB);
    std::printf("[race] run%d expected=%s got=%s matchesExpected=%s\n",r,
                fill==0x11111111u?"A":"B",
                matchesA?"A":(matchesB?"B":" other"),
                (fill==0x11111111u?matchesA:matchesB)?"YES":"NO (ASYNC/LAGGING!)");
  }
}

// ------- (i) large reductions used by gated_residual / final head ----------- //
void probe_reduce(){
  const int HC=4,H=2560,HCH=HC*H;
  std::printf("\n=== reductions: mx::mean(square(x),-1) over %d / %d elems ===\n",H,HCH);
  for(mx::Dtype dt:{mx::float32,mx::bfloat16}){
    char tag[40];std::snprintf(tag,sizeof(tag),"mean_sq-%s",dt==mx::float32?"f32":"bf16");
    auto xf=rand_floats((size_t)HC*H,81,2.0f);
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array x=make_array(xf,mx::Shape{1,1,HC,H},dt);
      mx::array y=mx::astype(x,mx::float32);
      mx::array m=mx::mean(mx::square(y),-1,true);
      mx::array out=mx::reshape(m,mx::Shape{-1});
      std::vector<float> cur=to_host(out);
      if(r==0)ref=cur;
      report(tag,r,hash_vec(cur),cur,ref);
    }
  }
  // mean over the hc axis (size 4) like final_head's `mixed`
  auto xf=rand_floats((size_t)HC*H,82,1.0f);
  std::vector<float> ref;
  for(int r=0;r<5;++r){
    mx::array x=make_array(xf,mx::Shape{1,1,HC,H},mx::bfloat16);
    mx::array out=mx::mean(x,-2,false);
    std::vector<float> cur=to_host(out);if(r==0)ref=cur;report("mean_hc",r,hash_vec(cur),cur,ref);
  }
}

// ------- (h) GDN sub-ops not yet isolated (fast rms_norm, gates, norm) ------- //
void probe_gdnsub(){
  const int B=1,T=1,HV=48,DV=128,DK=128;
  std::printf("\n=== GDN sub-ops: fast::rms_norm / rms_norm_gated / norm_qk / g,beta ===\n");
  auto xh=rand_floats((size_t)B*T*HV*DV,71,1.0f);
  auto wh=rand_bf16_bits((size_t)DV,72);
  auto zh=rand_floats((size_t)B*T*HV*DV,73,1.0f);
  auto qh=rand_floats((size_t)B*T*HV*DK,74,1.0f);
  auto kh=rand_floats((size_t)B*T*HV*DK,75,1.0f);
  auto ah=rand_floats((size_t)B*T*HV,76,1.0f);
  auto bh=rand_floats((size_t)B*T*HV,77,1.0f);
  auto alh=rand_floats((size_t)HV,78,0.5f);
  auto dbh=rand_floats((size_t)HV,79,0.1f);

  {
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array x(xh.data(),mx::Shape{B,T,HV,DV},mx::float32);
      mx::array w(wh.data(),mx::Shape{DV},mx::bfloat16);
      mx::array out=mx::fast::rms_norm(x,w,1e-6f);
      std::vector<float> cur=to_host(out);if(r==0)ref=cur;report("fast_rmsnorm",r,hash_vec(cur),cur,ref);
    }
  }
  {
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array x(xh.data(),mx::Shape{B,T,HV,DV},mx::bfloat16);
      mx::array w(wh.data(),mx::Shape{DV},mx::bfloat16);
      mx::array g(zh.data(),mx::Shape{B,T,HV,DV},mx::bfloat16);
      mx::array out=rms_norm_gated(x,w,g,1e-6f,true);
      std::vector<float> cur=to_host(out);if(r==0)ref=cur;report("rms_norm_gated",r,hash_vec(cur),cur,ref);
    }
  }
  {
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array q(qh.data(),mx::Shape{B,T,HV,DK},mx::float32);
      mx::array k(kh.data(),mx::Shape{B,T,HV,DK},mx::float32);
      auto qk=gdn_normalize_qk(q,k);
      std::vector<float> cq=to_host(qk.first),ck=to_host(qk.second);
      std::vector<float> cur=cq;cur.insert(cur.end(),ck.begin(),ck.end());
      if(r==0)ref=cur;report("gdn_norm_qk",r,hash_vec(cur),cur,ref);
    }
  }
  {
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array al(alh.data(),mx::Shape{HV},mx::float32);
      mx::array a(ah.data(),mx::Shape{B,T,HV},mx::float32);
      mx::array b(bh.data(),mx::Shape{B,T,HV},mx::float32);
      mx::array db(dbh.data(),mx::Shape{HV},mx::float32);
      auto gb=compute_g_beta(al,a,b,db);
      std::vector<float> cg=to_host(gb.first),cb=to_host(gb.second);
      std::vector<float> cur=cg;cur.insert(cur.end(),cb.begin(),cb.end());
      if(r==0)ref=cur;report("gdn_g_beta",r,hash_vec(cur),cur,ref);
    }
  }
}

// ------- (l) SDPA with the generator's exact settings / short KV cache ----- //
void probe_sdpa2(){
  const int B=1,HQ=24,HKV=2,D=256;
  std::printf("\n=== SDPA generator settings: force_fused=false, mask=causal, short L ===\n");
  auto qh=rand_floats((size_t)B*HQ*1*D,301,1.0f);
  float scale=1.0f/std::sqrt((float)D);
  for(int L:{1,2,3,4,8,16}){
    auto kh=rand_floats((size_t)B*HKV*L*D,302+L,1.0f);
    auto vh=rand_floats((size_t)B*HKV*L*D,303+L,1.0f);
    for(mx::Dtype dt:{mx::bfloat16,mx::float32}){
      char tag[48];std::snprintf(tag,sizeof(tag),"sdpa-L%d-%s",L,dt==mx::bfloat16?"bf16":"f32");
      std::vector<float> ref;
      for(int r=0;r<5;++r){
        mx::array q(qh.data(),mx::Shape{B,HQ,1,D},dt);
        mx::array k(kh.data(),mx::Shape{B,HKV,L,D},dt);
        mx::array v(vh.data(),mx::Shape{B,HKV,L,D},dt);
        mx::array out=mx::fast::scaled_dot_product_attention(q,k,v,scale,"causal",std::nullopt,std::nullopt,false);
        std::vector<float> cur=to_host(out);if(r==0)ref=cur;
        report(tag,r,hash_vec(cur),cur,ref);
      }
    }
  }
}

// ------- (k) GDN projection matmul shapes (M=1 GEMV, various N) ------------- //
void probe_matmul(){
  std::printf("\n=== mx::matmul GEMV shapes used by GDN projections ===\n");
  struct S{int K,N;const char* name;};
  const S shapes[]={{2560,10240,"in_proj_qkv"},{2560,6144,"in_proj_z"},
                    {2560,48,"in_proj_b"},{2560,48,"in_proj_a"},
                    {6144,2560,"out_proj"},{2560,128,"norm-ish"}};
  for(const S& s:shapes){
    auto xf=rand_floats(s.K,91,0.02f);
    auto wf=rand_floats((size_t)s.N*s.K,92,0.02f);
    for(mx::Dtype dt:{mx::float32,mx::bfloat16}){
      char tag[64];std::snprintf(tag,sizeof(tag),"%s-%s",s.name,dt==mx::float32?"f32":"bf16");
      std::vector<float> ref;
      for(int r=0;r<5;++r){
        mx::array x=make_array(xf,mx::Shape{1,1,s.K},dt);
        mx::array w=make_array(wf,mx::Shape{s.N,s.K},dt);
        mx::array out=mx::matmul(x,mx::transpose(w));
        std::vector<float> cur=to_host(out);if(r==0)ref=cur;
        report(tag,r,hash_vec(cur),cur,ref);
      }
    }
  }
}

// ---------- (j) checkpoint weight hashing (generator's exact load path) ----- //
namespace wprobe {
using Config = std::map<std::string, std::string>;

Config read_config(const std::string& path) {
  Config cfg;std::ifstream f(path);std::string line;
  while(std::getline(f,line)){
    if(line.empty()||line[0]=='#')continue;
    auto eq=line.find('=');if(eq==std::string::npos)continue;
    std::string v=line.substr(eq+1);
    while(!v.empty()&&(v.back()=='\r'||v.back()=='\n'))v.pop_back();
    cfg[line.substr(0,eq)]=v;
  }
  return cfg;
}
int get_int(const Config& c,const std::string& k,int d=0){auto i=c.find(k);return i==c.end()?d:std::stoi(i->second);}

struct Loader {
  std::string dir;int gs,bits;
  std::map<std::string,std::string> shard_of;
  std::map<std::string,std::unordered_map<std::string,mx::array>> shards;
  Loader(const std::string& d,const Config& c,int g,int b):dir(d),gs(g),bits(b){
    for(const auto& [k,v]:c) if(k.rfind("shard.",0)==0) shard_of[k.substr(6)]=v;
  }
  mx::array raw(const std::string& name){
    auto it=shard_of.find(name);
    if(it==shard_of.end()) throw std::runtime_error("no shard for "+name);
    auto sit=shards.find(it->second);
    if(sit==shards.end()){
      auto [w,m]=mx::load_safetensors(dir+"/"+it->second);
      sit=shards.emplace(it->second,std::move(w)).first;
      std::fprintf(stderr,"[wprobe] loaded shard %s (%zu tensors)\n",it->second.c_str(),sit->second.size());
    }
    auto t=sit->second.find(name);
    if(t==sit->second.end()) throw std::runtime_error("missing tensor "+name);
    return t->second;
  }
  static mx::array as_bf16(mx::array a){
    if(static_cast<mx::Dtype::Val>(a.dtype())==mx::uint16) return mx::view(a,mx::bfloat16);
    return a;
  }
  mx::array dense(const std::string& name){
    const std::string base=name.size()>7&&name.compare(name.size()-7,7,".weight")==0
        ?name.substr(0,name.size()-7):name;
    if(shard_of.count(base+".scales")==0) return raw(name);
    mx::array w=raw(base+".weight");
    mx::array s=as_bf16(raw(base+".scales"));
    mx::array b=as_bf16(raw(base+".biases"));
    mx::array out=mx::dequantize(w,s,b,gs,bits,"affine");
    mx::eval(out);
    return out;
  }
};

std::string hash_array(const mx::array& a){
  mx::array f=mx::reshape(mx::astype(a,mx::float32),mx::Shape{-1});
  mx::eval(f);
  const float* p=f.data<float>();
  return sha256_hex(p,static_cast<size_t>(f.size())*sizeof(float));
}

void probe_weights(const std::string& ckpt,const std::string& cfgpath){
  Config c=read_config(cfgpath);
  int gs=get_int(c,"group_size",32),bits=get_int(c,"bits",4);
  int H=get_int(c,"hidden_size"),kc=get_int(c,"key_dim"),vc=get_int(c,"value_dim");
  int nkh=get_int(c,"num_k_heads"),nvh=get_int(c,"num_v_heads");
  int dk=get_int(c,"head_k_dim"),dv=get_int(c,"head_v_dim"),convk=get_int(c,"conv_kernel");
  Loader L(ckpt,c,gs,bits);
  for(int layer:{0,5,6,7}){
    const std::string b="language_model.model.layers."+std::to_string(layer)+".";
    std::fprintf(stderr,"[wprobe] === layer %d ===\n",layer);
    auto emit=[&](const char* tag,mx::array a){
      std::printf("L%d %-22s %s\n",layer,tag,hash_array(a).c_str());
    };
    const std::string q=b+"linear_attn.";
    if(L.shard_of.count(q+"in_proj_qkv.weight")){
      emit("in_proj_qkv",L.dense(q+"in_proj_qkv"));
      emit("in_proj_z",L.dense(q+"in_proj_z"));
      emit("in_proj_b",L.dense(q+"in_proj_b"));
      emit("in_proj_a",L.dense(q+"in_proj_a"));
      emit("conv1d.weight",L.dense(q+"conv1d.weight"));
      emit("A_log",L.dense(q+"A_log"));
      emit("dt_bias",L.dense(q+"dt_bias"));
      emit("norm.weight",L.dense(q+"norm.weight"));
      emit("out_proj",L.dense(q+"out_proj"));
    }else{
      const std::string a=b+"self_attn.";
      emit("q_proj(fullattn)",L.dense(a+"q_proj"));
      emit("k_proj(fullattn)",L.dense(a+"k_proj"));
    }
  }
  // Step 3: hash the GDN cache-state initialization bytes (mx::zeros shapes).
  int cdim=2*kc+vc;
  mx::array cz=mx::zeros(mx::Shape{1,convk-1,cdim},mx::float32);
  mx::array rz=mx::zeros(mx::Shape{1,nvh,dv,dk},mx::float32);
  std::printf("CACHE conv_state  %s\n",hash_array(cz).c_str());
  std::printf("CACHE rec_state   %s\n",hash_array(rz).c_str());
  (void)H;
}
}  // namespace wprobe

// --------------------- (f) NaN / Inf stress on the ops ---------------------- //
void probe_nan(){
  const int N=248320,K=2560,GS=32,BITS=4,SN=K/GS,WPACK=K/8;
  std::printf("\n=== NaN/Inf stress: qmm + argmax on poisoned inputs ===\n");
  // Packed weights with a few NaN-producing scale rows.
  std::vector<uint32_t> wbits((size_t)N*WPACK);
  {std::mt19937_64 rng(12345);for(auto& x:wbits)x=uint32_t(rng());}
  auto sbits=rand_bf16_bits((size_t)N*SN,777);
  auto bbits=rand_bf16_bits((size_t)N*SN,999);
  // poison ~1% of scale rows with bf16 +Inf / NaN patterns
  for(size_t i=0;i<sbits.size();i+=100){sbits[i]=0x7f80u;sbits[i+1]=0x7fc0u;}
  auto xf=rand_floats(K,20261117,0.02f);
  for(mx::Dtype dt:{mx::float32,mx::bfloat16}){
    char ntag[32];std::snprintf(ntag,sizeof(ntag),"qmm-nan-%s",dt==mx::float32?"f32":"bf16");
    std::vector<float> ref;
    for(int r=0;r<5;++r){
      mx::array x=make_array(xf,mx::Shape{1,1,K},dt);
      mx::array w(wbits.data(),mx::Shape{N,WPACK},mx::uint32);
      mx::array s(sbits.data(),mx::Shape{N,SN},mx::bfloat16);
      mx::array b(bbits.data(),mx::Shape{N,SN},mx::bfloat16);
      mx::array out=mx::quantized_matmul(x,w,s,b,true,GS,BITS,"affine");
      std::vector<float> cur=to_host(out);
      if(r==0)ref=cur;
      report(ntag,r,hash_vec(cur),cur,ref);
    }
  }
  // argmax on a row containing NaNs at several positions
  std::printf("-- argmax with NaNs --\n");
  auto vh=rand_floats(4096,31,1.0f);
  for(size_t p=0;p<vh.size();p+=512)vh[p]=std::numeric_limits<float>::quiet_NaN();
  std::vector<float> ref;
  for(int r=0;r<5;++r){
    mx::array la(vh.data(),mx::Shape{1,4096},mx::float32);
    mx::array am=sample_greedy(la);mx::eval(am);
    std::vector<float> cur=to_host(am);
    if(r==0){ref=cur;std::printf("argmax-nan run0 idx=%d\n",(int)cur[0]);}
    else std::printf("argmax-nan run%d idx=%d %s\n",r,(int)cur[0],(cur[0]==ref[0]?"IDENTICAL":"VARIES"));
  }
}

// --------------------- (d) router argpartition + argmax --------------------- //
void probe_router(){
  const int E=512,K=10,V=248320;
  std::printf("\n=== router: softmax+argpartition(E=%d,k=%d) and argmax(V=%d) ===\n",E,K,V);

  // variant 2 = EXACT ties for 64 experts (argpartition must break ties).
  for(int variant=0;variant<3;++variant){
    auto gh=rand_floats(E,42+variant, variant?0.001f:3.0f);
    if(variant==2){for(int i=0;i<64;++i)gh[i]=1.0f; /* exact ties */}
    std::vector<float> refi,refs,refa;
    for(int r=0;r<5;++r){
      mx::array lg(gh.data(),mx::Shape{1,E},mx::float32);
      RouterOutput ro=router(lg,K,0.0f);
      mx::array vi=mx::astype(ro.indices,mx::int32),vs=ro.scores;
      mx::eval(vi);mx::eval(vs);
      std::vector<float> iv=to_host(vi),sv=to_host(vs);
      // argmax over a big vocab with near-ties
      auto vh=rand_floats(V,99+variant,variant?0.0005f:2.0f);
      mx::array la(vh.data(),mx::Shape{1,V},mx::float32);
      mx::array am=sample_greedy(la);mx::eval(am);
      std::vector<float> av=to_host(am);
      char rtag[32];std::snprintf(rtag,sizeof(rtag),"router-v%d",variant);
      std::printf("[%s] run%d sha256=%s\n",rtag,r,hash_vec(iv).c_str());
      if(r==0){refi=iv;refs=sv;refa=av;
        std::printf(" variant%d run0 idx0=%d idx9=%d argmax=%d idxsha=%s\n",variant,
                    (int)iv[0],(int)iv[K-1],(int)av[0],hash_vec(iv).c_str());}
      else{
        size_t di=0,ds=0;float mi=0,ms=0;
        for(size_t i=0;i<iv.size();++i){if(iv[i]!=refi[i])++di;mi=std::max(mi,std::fabs(iv[i]-refi[i]));}
        for(size_t i=0;i<sv.size();++i){if(sv[i]!=refs[i])++ds;ms=std::max(ms,std::fabs(sv[i]-refs[i]));}
        std::printf(" variant%d run%d idxdiff=%zu scoresdiff=%zu argmax=%d(%s) %s\n",
                    variant,r,di,ds,(int)av[0],(av[0]==refa[0]?"same":"CHANGED"),
                    (di||ds||av[0]!=refa[0])?"VARIES":"IDENTICAL");
      }
    }
  }
}

// --------------------------- (c) fused SDPA -------------------------------- //
void probe_sdpa(){
  const int B=1,HQ=24,HKV=2,D=256,L=16;
  std::printf("\n=== fused SDPA: q[1,%d,1,%d] k/v[1,%d,%d,%d] causal ===\n",HQ,D,HKV,L,D);
  auto qh=rand_floats((size_t)B*HQ*1*D,11,1.0f);
  auto kh=rand_floats((size_t)B*HKV*L*D,12,1.0f);
  auto vh=rand_floats((size_t)B*HKV*L*D,13,1.0f);
  float scale=1.0f/std::sqrt((float)D);
  std::vector<float> ref;
  for(int r=0;r<5;++r){
    mx::array q(qh.data(),mx::Shape{B,HQ,1,D},mx::bfloat16);
    mx::array k(kh.data(),mx::Shape{B,HKV,L,D},mx::bfloat16);
    mx::array v(vh.data(),mx::Shape{B,HKV,L,D},mx::bfloat16);
    mx::array out=mx::fast::scaled_dot_product_attention(q,k,v,scale,"causal",std::nullopt,std::nullopt,/*force_fused=*/true);
    std::vector<float> cur=to_host(out);
    if(r==0)ref=cur;
    report("sdpa",r,hash_vec(cur),cur,ref);
  }
}
}  // namespace

int main(int argc,char** argv){
  std::string mode=argc>1?argv[1]:"all";
  std::printf("MLX determinism probe | mode=%s\n",mode.c_str());
  if(mode=="weights"){
    if(argc<4){std::fprintf(stderr,"usage: det_probe weights <checkpoint-dir> <config>\n");return 2;}
    wprobe::probe_weights(argv[2],argv[3]);
    return 0;
  }
  if(mode=="all"||mode=="qmm")  probe_qmm(false);
  if(mode=="all"||mode=="dqmm") probe_qmm(true);
  if(mode=="all"||mode=="gdn")  probe_gdn();
  if(mode=="all"||mode=="sdpa") probe_sdpa();
  if(mode=="all"||mode=="router") probe_router();
  if(mode=="all"||mode=="gqmm") probe_gqmm();
  if(mode=="all"||mode=="nan") probe_nan();
  if(mode=="all"||mode=="race") probe_race();
  if(mode=="all"||mode=="gdnsub") probe_gdnsub();
  if(mode=="all"||mode=="reduce") probe_reduce();
  if(mode=="all"||mode=="matmul") probe_matmul();
  if(mode=="all"||mode=="sdpa2") probe_sdpa2();
  return 0;
}
