//! CPU scalar adapters handle recurrence/merge and QK/PV, not GPU MMA or scheduling; GPU chunk A/B remains required.
const std = @import("std");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 CUDA attention source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_cuda_attention_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
fn between(source: []const u8, first: []const u8, last: []const u8) ![]const u8 {
    const start = std.mem.indexOf(u8, source, first) orelse return error.CudaOnlineSource;
    const end = std.mem.indexOfPos(u8, source, start + first.len, last) orelse return error.CudaOnlineSource;
    return source[start..end];
}
fn restoreTile(a: std.mem.Allocator, source: []const u8) ![]u8 {
    // Exact inverse of KP10, including the downstream add/multiply/FMA choices in the original recurrence.
    const edits = [_][2][]const u8{
        .{ "float next_m[2], alpha[2], rows[2]; bool same[2];", "float next_m[2], alpha[2], rows[2];" },
        .{ "same[h] = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(next_m[h]) && next_m[h] == m[h]; alpha[h] = same[h] ? 1.0f : (active ? (m[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m[h], next_m[h]))) : 1.0f);", "alpha[h] = active ? (m[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m[h], next_m[h]))) : 1.0f;" },
        .{ "l[h] = (same[h] && isfinite(__fadd_rn(l[h], sum))) ? __fadd_rn(l[h], sum) : __fmaf_rn(l[h], alpha[h], sum);", "l[h] = __fmaf_rn(l[h], alpha[h], sum);" },
        .{ "for (int e = 0; e < 4; ++e) if (!same[e >> 1] || !isfinite(o[q][e])) o[q][e] = __fmul_rn(o[q][e], alpha[e >> 1]);", "for (int e = 0; e < 4; ++e) o[q][e] = __fmul_rn(o[q][e], alpha[e >> 1]);" },
        .{ "const bool same = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(next) && next == mm[h]; const float a = same ? 1.0f : (active ? (mm[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(mm[h], next))) : 1.0f);", "const float a = active ? (mm[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(mm[h], next))) : 1.0f;" },
        .{ "for (int e = 0; e < 2; ++e) oo[FUSE ? q : 0][2 * h + e] = (same && isfinite(__fadd_rn(oo[FUSE ? q : 0][2 * h + e], __fmul_rn(o[q][2 * h + e], bb)))) ? __fadd_rn(oo[FUSE ? q : 0][2 * h + e], __fmul_rn(o[q][2 * h + e], bb)) : __fmaf_rn(o[q][2 * h + e], bb, __fmul_rn(oo[FUSE ? q : 0][2 * h + e], a));", "for (int e = 0; e < 2; ++e) oo[FUSE ? q : 0][2 * h + e] = __fmaf_rn(o[q][2 * h + e], bb, __fmul_rn(oo[FUSE ? q : 0][2 * h + e], a));" },
        .{ "ll[h] = (same && isfinite(__fadd_rn(ll[h], __fmul_rn(l[h], bb)))) ? __fadd_rn(ll[h], __fmul_rn(l[h], bb)) : __fmaf_rn(l[h], bb, __fmul_rn(ll[h], a));", "ll[h] = __fmaf_rn(l[h], bb, __fmul_rn(ll[h], a));" },
    };
    var out = source;
    for (edits) |pair| out = try edit.once(a, out, pair[0], pair[1]);
    return edit.once(a, out, "static void online_guarded(", "static void online_legacy(");
}
fn restoreMerge(a: std.mem.Allocator, source: []const u8) ![]u8 {
    const edits = [_][2][]const u8{
        .{ "const bool same = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(next) && next == m; const float a = same ? 1.0f : (active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f);", "const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;" },
        .{ "o = (same && isfinite(__fadd_rn(o, __fmul_rn(co, b)))) ? __fadd_rn(o, __fmul_rn(co, b)) : __fmaf_rn(co, b, __fmul_rn(o, a));", "o = __fmaf_rn(co, b, __fmul_rn(o, a));" },
        .{ "l = (same && isfinite(__fadd_rn(l, __fmul_rn(cl, b)))) ? __fadd_rn(l, __fmul_rn(cl, b)) : __fmaf_rn(cl, b, __fmul_rn(l, a));", "l = __fmaf_rn(cl, b, __fmul_rn(l, a));" },
    };
    var out = source;
    for (edits) |pair| out = try edit.once(a, out, pair[0], pair[1]);
    return edit.once(a, out, "fn_attn_merge_body(", "fn_attn_merge_body_legacy(");
}
fn onlineSource(a: std.mem.Allocator, chunk: []const u8) ![]const u8 {
    const probability = "#pragma unroll\n        for (int jt = 0; jt < JT; ++jt)\n#pragma unroll\n            for (int e = 0; e < 4; ++e) {\n                const int h = e >> 1;";
    const rescale = "#pragma unroll\n        for (int q = 0; q < 8; ++q)";
    const next = try between(chunk, "        float next_m[2], alpha[2], rows[2]; bool same[2];", probability);
    const probabilities = try between(chunk, probability, "#pragma unroll\n        for (int h = 0; h < 2; ++h) {\n            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync");
    const sums = try between(chunk, "#pragma unroll\n        for (int h = 0; h < 2; ++h) {\n            const int row = g + 8 * h;\n            const float sum =", rescale);
    const outputs = try between(chunk, rescale, "        for (int ks = 0; ks < TK / 16; ++ks) {");
    const fused = try between(chunk, "        for (int h = 0; h < 2; ++h) {\n            const bool active = l[h] > 0.0f;", "        continue;");
    const final = try between(chunk, "                    OUT[row + warp * 64 + q * 8 + 2 * t + e] =", "\n");
    return std.mem.concat(a, u8, &.{
        \\static void online_guarded(const float* scores,const uint16_t* values,int n,int chunk,int fused,float* po,float* pm,float* pl,float* parts,uint16_t* out){
        \\ constexpr bool FUSE=true;constexpr int QR=16,TK=32;struct Lane{float mm,ll,oo[8][4];};Lane fs[4][4]={};
        \\ for(int w=0;w<4;++w)for(int t=0;t<4;++t)fs[w][t].mm=-INFINITY;
        \\ const int chunks=(n+chunk-1)/chunk;
        \\ for(int cc=0;cc<chunks;++cc){
        \\  float m[2]={-INFINITY,-INFINITY},l[2]={0.0f,0.0f},lanes[4][4][8][4]={};
        \\  const int chunk_keys=min(chunk,n-cc*chunk);
        \\  for(int base=0;base<chunk_keys;base+=TK){
        \\   const int valid=min(TK,chunk_keys-base);float tile_s[TK],s[1][4];bfloat ps[QR][TK+8]={};
        \\   for(int j=0;j<TK;++j)tile_s[j]=j<valid?scores[cc*chunk+base+j]:-INFINITY;
        \\   float tm=-INFINITY;for(int j=0;j<TK;++j)tm=fmaxf(tm,tile_s[j]);
        \\   float red_m[4][QR];for(int w=0;w<4;++w)for(int row=0;row<QR;++row)red_m[w][row]=row==0?tm:-INFINITY;
        \\   const int g=0,warp=0;
        \\
        ,
        next,
        \\   for(int pair=0;pair<TK/2;++pair){const int t=pair;s[0][0]=tile_s[2*pair];s[0][1]=tile_s[2*pair+1];s[0][2]=-INFINITY;s[0][3]=-INFINITY;
        \\
        ,
        probabilities,
        \\   }
        \\   float red_l[4][QR]={};red_l[0][0]=rows[0];red_l[0][8]=rows[1];
        \\
        ,
        sums,
        \\   for(int w=0;w<4;++w)for(int t=0;t<4;++t){
        \\    float (*o)[4]=lanes[w][t];
        \\
        ,
        outputs,
        \\    for(int q=0;q<8;++q)for(int e=0;e<2;++e){const int d=w*64+q*8+2*t+e;
        \\     for(int j=0;j<TK;++j){const float v=j<valid?float(reinterpret_cast<const bfloat*>(values)[(cc*chunk+base+j)*HD+d]):0.0f;o[q][e]=__fmaf_rn(float(ps[g][j]),v,o[q][e]);}
        \\    }
        \\   }
        \\  }
        \\  if(fused){
        \\   for(int w=0;w<4;++w)for(int t=0;t<4;++t){Lane& lane=fs[w][t];float mm[2]={lane.mm,-INFINITY},ll[2]={lane.ll,0.0f},oo[8][4];std::memcpy(oo,lane.oo,sizeof(oo));float (*o)[4]=lanes[w][t];
        \\
        ,
        fused,
        \\    lane.mm=mm[0];lane.ll=ll[0];std::memcpy(lane.oo,oo,sizeof(oo));
        \\   }
        \\  }
        \\  for(int w=0;w<4;++w)for(int t=0;t<4;++t)for(int q=0;q<8;++q)for(int e=0;e<2;++e){const int d=w*64+q*8+2*t+e;parts[cc*(HD+2)+d]=fused?fs[w][t].oo[q][e]:lanes[w][t][q][e];}
        \\  parts[cc*(HD+2)+HD]=fused?fs[0][0].mm:m[0];parts[cc*(HD+2)+HD+1]=fused?fs[0][0].ll:l[0];
        \\  if(!fused){for(int d=0;d<HD;++d)po[cc*HD+d]=parts[cc*(HD+2)+d];pm[cc]=m[0];pl[cc]=l[0];}
        \\ }
        \\ if(fused)for(int warp=0;warp<4;++warp)for(int t=0;t<4;++t){Lane& lane=fs[warp][t];float ll[2]={lane.ll,0.0f},oo[8][4];std::memcpy(oo,lane.oo,sizeof(oo));__nv_bfloat16* OUT=reinterpret_cast<__nv_bfloat16*>(out);const long long row=0;const int h=0;
        \\  for(int q=0;q<8;++q)for(int e=0;e<2;++e){
        \\
        ,
        final,
        \\  }
        \\ }
        \\}
        \\
        ,
    });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const text = @embedFile("../kernels/cuda/flashnext_attn.cu");
    if (std.mem.indexOf(u8, text, "constexpr int HD = 256;") == null or std.mem.indexOf(u8, text, "constexpr int TK = 32;") == null) return error.CudaGeometry;
    const start = std.mem.indexOf(u8, text, "                const float p = s[jt][e]") orelse return error.CudaProbability;
    const last = std.mem.indexOfPos(u8, text, start, "ps[g + 8 * h]") orelse return error.CudaProbability;
    const end = (std.mem.indexOfScalarPos(u8, text, last, ';') orelse return error.CudaProbability) + 1;
    const store = text[start..end];
    if (std.mem.count(u8, store, "__float2bfloat16_rn(p)") != 1 or std.mem.indexOf(u8, store, "rows[h] = __fadd_rn(rows[h], p);") == null) return error.CudaProbability;
    // #1739 moved the merge into a device body the kernel calls; the host instrument runs both.
    const body = try edit.once(a, try edit.function(text, "__device__ __forceinline__ void fn_attn_merge_body("), "__device__ __forceinline__ ", "inline ");
    var merge = try edit.function(text, "void __launch_bounds__(256) fn_attn_merge(");
    merge = try edit.once(a, merge, "void __launch_bounds__(256)", "void");
    merge = try edit.once(a, merge, "__global__ ", "");
    const chunk_text = try edit.once(a, text, "Pg pg = {}) {", "Pg pg) {");
    const chunk = try edit.function(chunk_text, "__device__ __forceinline__ void attn_chunks(");
    const guarded_online = try onlineSource(a, chunk);
    const legacy_online = try restoreTile(a, guarded_online);
    const legacy_merge = try restoreMerge(a, body);
    return std.mem.concat(a, u8, &.{
        \\#ifndef TF_CUDA_SOFTMAX_IDENTITY_GUARD
        \\#define TF_CUDA_SOFTMAX_IDENTITY_GUARD 1
        \\#endif
        \\#include "hc_native_support.h"
        \\namespace ca {
        \\constexpr int HD=256,JT=1;static uint3 blockIdx,threadIdx;
        \\inline float tf_exp(float x){return expf(x); }
        \\
        ,
        \\static bfloat probability(float score,float maximum,float& sum){
        \\ const int jt=0,e=0,h=0,g=0,warp=0,t=0;const float s[1][4]={{score}},next_m[2]={maximum};
        \\ float rows[2]={};bfloat ps[16][32];
        \\
        ,
        store,
        "\n sum=rows[0];return ps[0][0];}\n",
        guarded_online,
        "\n",
        legacy_online,
        "\n",
        body,
        "\n",
        legacy_merge,
        "\n",
        merge,
        "\n",
        \\static void execute(const float* scores,const uint16_t* values,int n,int chunk,int fused,float* parts,uint16_t* out,bool legacy){
        \\ if(n<0||chunk<1||chunk>2048)std::abort();const int chunks=(n+chunk-1)/chunk,alloc=chunks?chunks:1;
        \\ float* po=fused?nullptr:static_cast<float*>(std::malloc(sizeof(float)*HD*alloc));float* pm=fused?nullptr:static_cast<float*>(std::malloc(sizeof(float)*alloc));float* pl=fused?nullptr:static_cast<float*>(std::malloc(sizeof(float)*alloc));
        \\ if(!fused&&(!po||!pm||!pl))std::abort();
        \\ if(legacy)online_legacy(scores,values,n,chunk,fused,po,pm,pl,parts,out);else online_guarded(scores,values,n,chunk,fused,po,pm,pl,parts,out);
        \\ if(!fused){const int pos=n-1;blockIdx={0,0,0};for(int d=0;d<HD;++d){threadIdx={uint(d),0,0};if(legacy)fn_attn_merge_body_legacy(po,pm,pl,&pos,reinterpret_cast<bfloat*>(out),nullptr,nullptr,1,1,1,chunk,chunks,0);else fn_attn_merge_body(po,pm,pl,&pos,reinterpret_cast<bfloat*>(out),nullptr,nullptr,1,1,1,chunk,chunks,0);}}
        \\ std::free(po);std::free(pm);std::free(pl);
        \\}
        \\static void served(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,const int32_t* nk,const int32_t* spr,int rows,int keys,int fused,float* old_parts,float* new_parts,uint16_t* old_out,uint16_t* new_out){
        \\ if(rows<1||keys<1||keys>2048)std::abort();const int chunks=(keys+511)/512;
        \\ for(int r=0;r<rows;++r)for(int h=0;h<24;++h){const int n=spr[r]?nk[r]:keys;if(n<1||n>keys)std::abort();float* scores=static_cast<float*>(std::malloc(sizeof(float)*n));uint16_t* values=static_cast<uint16_t*>(std::malloc(sizeof(uint16_t)*HD*n));if(!scores||!values)std::abort();
        \\  for(int i=0;i<n;++i){const int source=spr[r]?ids[r*keys+i]:i;float dot=0;for(int d=0;d<HD;++d)dot=__fmaf_rn(float(reinterpret_cast<const bfloat*>(q)[(r*24+h)*HD+d]),float(reinterpret_cast<const bfloat*>(k)[(source*4+h/6)*HD+d]),dot);scores[i]=__fmul_rn(dot,0.0625f);for(int d=0;d<HD;++d)values[i*HD+d]=v[(source*4+h/6)*HD+d];}
        \\  const long long at=static_cast<long long>(r*24+h)*chunks;execute(scores,values,n,512,fused,old_parts+at*(HD+2),old_out+(r*24+h)*HD,true);execute(scores,values,n,512,fused,new_parts+at*(HD+2),new_out+(r*24+h)*HD,false);std::free(scores);std::free(values);
        \\ }
        \\}
        \\
        \\static void run(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output,int control){
        \\ if(count<1||count>32)std::abort();float scores[32],sum=0,maximum=-INFINITY; bfloat weights[32];
        \\ for(int i=0;i<count;++i){float dot=0;for(int d=0;d<256;++d)dot=fmaf(float(reinterpret_cast<const bfloat*>(q)[d]),float(reinterpret_cast<const bfloat*>(k)[ids[i]*256+d]),dot);scores[i]=dot*0.0625f;maximum=fmaxf(maximum,scores[i]);}
        \\ for(int i=0;i<count;++i){float term;weights[i]=probability(scores[i],maximum,term);sum+=term;}
        \\ if(control)for(int i=0;i<count;++i)weights[i]=bfloat(expf(scores[i]-maximum)/sum);
        \\ float po[256];for(int d=0;d<256;++d){po[d]=0;for(int i=0;i<count;++i)po[d]=fmaf(float(weights[i]),float(reinterpret_cast<const bfloat*>(v)[ids[i]*256+d]),po[d]);}
        \\ const float pm=maximum,pl=control?1.0f:sum;const int pos=count-1;
        \\ blockIdx={0,0,0};for(int d=0;d<256;++d){threadIdx={uint(d),0,0};fn_attn_merge(po,&pm,&pl,&pos,reinterpret_cast<bfloat*>(output),nullptr,nullptr,1,1,1,512,1,0);}
        \\}
        \\}
        \\extern "C" void tf_q4_cuda_node_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output,int control){ca::run(q,k,v,ids,count,output,control);}
        \\extern "C" void tf_q4_cuda_online_ab(const float* scores,const uint16_t* values,int count,int chunk,int fused,float* old_parts,float* new_parts,uint16_t* old_out,uint16_t* new_out){ca::execute(scores,values,count,chunk,fused,old_parts,old_out,true);ca::execute(scores,values,count,chunk,fused,new_parts,new_out,false);}
        \\extern "C" void tf_q4_cuda_served_ab(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,const int32_t* nk,const int32_t* spr,int rows,int keys,int fused,float* old_parts,float* new_parts,uint16_t* old_out,uint16_t* new_out){ca::served(q,k,v,ids,nk,spr,rows,keys,fused,old_parts,new_parts,old_out,new_out);}
        \\
        ,
    });
}
