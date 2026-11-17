#include <mlx/mlx.h>
#include <cstdio>
namespace mx = mlx::core;
int main(){
  int n=25;
  auto a=mx::arange(n);
  auto b=mx::astype(mx::arange(n), mx::int32);
  auto c=mx::arange(n, mx::int32);
  auto d=mx::astype(a,mx::int32);
  mx::eval(a,b,c,d);
  auto pr=[](const char* nm, const mx::array& x){
    int k=(int)x.dtype(); auto f=mx::astype(x,mx::float32); mx::eval(f);
    printf("%s dtype=%d size=%zu first=",nm,k,x.size());
    const float* p=f.data<float>(); for(int i=0;i<6;i++) printf("%g ",p[i]); printf("\n");
  };
  pr("arange(n)",a); pr("astype(arange,int32)",b); pr("arange(n,int32)",c); pr("astype(a,int32)",d);
  auto e=mx::astype(mx::arange(n),mx::int32)/4;
  mx::eval(e); pr("astype(arange,int32)/4",e);
  return 0;
}
