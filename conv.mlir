// REPRO: mlir-opt conv.mlir --test-transform-dialect-interpreter   --canonicalize --eliminate-empty-tensors --empty-tensor-to-alloc-tensor  -one-shot-bufferize='allow-return-allocs bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'

// N=5 OH=80 OW=100 F=C=128 KH=KW=3

!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>

func.func @conv(
    %input: !tinput {bufferization.writable = false},
    %filter: !tfilter {bufferization.writable = false},
    %bias: !tbias {bufferization.writable = false},
    %bias_init: !toutput,
    %output: !toutput {bufferization.writable = true,
                       bufferization.buffer_layout = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       bufferization.access = "read-write"}) -> !toutput 
  attributes { llvm.emit_c_interface }                       
{

  // Bias.
  // %bias_init = tensor.empty() : !toutput
  %biased = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel"],
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
  } ins(%bias : !tbias) outs(%bias_init : !toutput) {
  ^bb0(%arg0: f32, %arg2: f32):
    linalg.yield %arg0 : f32
  } -> !toutput

  // Convolution proper.
  %convolved = linalg.conv_2d_nhwc_fhwc ins(%input, %filter: !tinput, !tfilter) outs(%biased : !toutput) -> !toutput
  
  // ReLU.
  %c0 = arith.constant 0.0 : f32
  %relued = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel"],
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
  } ins(%c0, %convolved : f32, !toutput) outs(%output : !toutput) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.maxf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> !toutput


  return %relued : !toutput
}

// func.func private @printF32(tensor<*xf32>)

// func.func @main() {
//   %input_init = tensor.empty() : !tinput
//   %c1 = arith.constant 1.0 : f32
//   %input = linalg.fill ins(%c1 : f32) outs(%input_init : !tinput) -> !tinput

//   %bias_init = tensor.empty() : !tbias
//   %c01 = arith.constant 0.1 : f32
//   %bias = linalg.fill ins(%c01 : f32) outs(%bias_init : !tbias) -> !tbias

//   %filter_init = tensor.empty() : !tfilter
//   %c2 = arith.constant 2.0 : f32
//   %filter = linalg.fill ins(%c2 : f32) outs(%filter_init : !tfilter) -> !tfilter

//   %output = tensor.empty() : !toutput
//   %bias_init2 = tensor.empty() : !toutput

//   %result = func.call @conv(%input, %filter, %bias, %bias_init2, %output) : (!tinput, !tfilter, !tbias, !toutput, !toutput) -> !toutput
//   %result_unranked = tensor.cast %result : !toutput to tensor<*xf32>
//   func.call @printF32(%result_unranked) : (tensor<*xf32>) -> ()

//   return
// }

// transform.sequence failures(propagate) {
// ^bb0(%arg0: !transform.any_op):
//   %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg0 : (!transform.any_op) -> !pdl.operation

//   %t1, %loops:7 = transform.structured.tile_to_scf_for %0 
//             // N,  F, OH, OW,  C, KH, KW   /// this is currently a wrong order based on nchw_fchw
//             //[1,  4,  1,  6,  1,  1,  3]  /// the correct order is::: nhwc_fhwc ->  N OH OW F KH KW C
//             // N  OH  OW   F  KH  KW   C 
//             //[4,  1,  6,  4,  1,  3,  1]
//               [5,  1, 10,  4,  1,  3,  1]
//     //{ interchange = [1, 5, 6, 0, 4, 2, 3] }  // F, KH, KW, N, C, OH, OW 
//     { interchange = [3, 4, 5, 0, 6, 1, 2] }  

//     // loop order:      F KW C N KH OH OW
//     // expected  :      F KH KW N C OH OW
//     // upd1 loop order: OH KW C N KW OW F
//                 //       1  5 6 0  4 2  3    // iterator order (assuming non-alt): N OH OW F KW KH C

//   %t2, %loops2:7 = transform.structured.tile_to_scf_for %t1
//         //        N,  F, OH, OW,  C, KH, KW  // this is currently a wrong order 
//         //       [1,  4,  1,  6,  1,  1,  3]
//                // N  OH  OW   F  KH  KW   C 
//                  [1,  1, 10,  4,  1,  3,  1]

//   %d = transform.structured.decompose %t2
//   // transform.structured.generalize %d

//   %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !pdl.operation
//   transform.structured.vectorize %f

//   transform.yield
// }

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %conv = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg0 : (!transform.any_op) -> !pdl.operation
  %1 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !pdl.operation
  %bias, %relu = transform.split_handles %1 in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

  // transform.test_print_remark_at_operand %conv, "convolution" : !pdl.operation
  // transform.test_print_remark_at_operand %bias, "bias" : !pdl.operation
  // transform.test_print_remark_at_operand %relu, "relu" : !pdl.operation

  
  // Order of dimensions in the output N OH OW F

  // Split C is actually tiling only F (name mismatch) dimension. vec * tile_w = 16 * 4 = 64.
  // We also want to parallelize the Co loop, so tile to forall.
  // %relu2, %co_loop = transform.structured.tile_to_scf_for %relu [0, 0, 0, 64]
  %co_loop, %relu2 = transform.structured.tile_to_forall_op %relu tile_sizes [0, 0, 0, 64] 

  // Split X is actually tiling only H (name mismatch) dimension. tile_h = 5.
  // But we cannot do it here because we want a specific loop order.
  // %relu3, %xo_loop = transform.structured.tile_to_scf_for %relu2 [0, 5, 0, 0]

  // Want loop order (innermost to outermost: ci, xi ,xo, y, n, co): co, n,  y, xo, xi, ci
  // After previous splits, the loop order is:                       co, xo, [n, xi, y, ci]
  //   where brackets indicate dimensions of the generic.
  // Additionally want to have y, n, co parallel, so create parallel loops for them by
  // tiling with tile size 1.

  %multi_loop, %relu3 = transform.structured.tile_to_forall_op %relu2 tile_sizes [1, 0, 1, 0] 
  // %relu3, %multi_loops:2 = transform.structured.tile_to_scf_for %relu2 [1, 0, 1, 0]

  // this would have given us co, xo, no, yo, [ni=1, xi, yi=1, ci] if we tiled X before
  // so we tile it after

  // %relu4, %xo_loop = transform.structured.tile_to_scf_for %relu3 [0, 5, 0, 0]
  %xo_loop, %relu4 = transform.structured.tile_to_forall_op %relu3 tile_sizes [0, 5]
  // Now we get the desired order:  co, n, y, xo, [ni=1, xi, yi=1, ci]

  // Vectorization currently applies to entire functions, so postpone.
  // TODO: reconsider and have vectorization more targeted.

  // Compute_at is actually fusion into the given loop.
  %conv2 = transform.structured.fuse_into_containing_op %conv into %co_loop

  %conv3 = transform.structured.fuse_into_containing_op %conv2 into %multi_loop
  %conv4 = transform.structured.fuse_into_containing_op %conv3 into %xo_loop

  // Also fuse the bias that we represent as a separate operation and Halide represents 
  // as the "pure" (as opposed to "update") part of the conv expression.
  %bias2 = transform.structured.fuse_into_containing_op %bias into %co_loop
  %bias3 = transform.structured.fuse_into_containing_op %bias2 into %multi_loop
  %bias4 = transform.structured.fuse_into_containing_op %bias3 into %xo_loop

  // We are not fusing inputs because we don't have operations for them.

  // Somehow get the loop order: [n, r.z, r.y, r.x, y, x, c] => [N, KW, KH, C, W, H, F]
  // Order of dimensions in the conv loop: N H W F KH KW C.
  // Cannot interchange here yet because will not be able to vectorize.
  // transform.structured.interchange %conv2 {iterator_interchange = array<i64: 0, 5, 4, 6, 2, 1, 3>}

  //

  // Now we should be ready to vectorize.
  // First, we need to decompose so vectorization is actually possible. For decomposition to work,
  // we need kh and oh to be 1
  // %conv3, %loops:2 = transform.structured.tile_to_scf_for %conv2 [0, 1, 0, 0, 1]

  // We also want to materialize other loops (F, H, W) so we can unroll them.
  // Order of dimensions in the conv loop: N H W F KH KW C.
  // Do the reoder as wanted above: [N, KW, KH, C, W, H, F]
  // %conv3, %loops:7 = transform.structured.tile_to_scf_for %conv2 [1, 1, 1, 1, 1, 1, 16]
  //   { interchange = [0, 5, 4, 3, 2, 1, 6] }
  %loops, %conv5 = transform.structured.tile_to_forall_op %conv4 tile_sizes [1, 1, 1, 1, 1, 1, 16]
  transform.structured.fuse_into_containing_op %bias4 into %loops

  transform.structured.decompose %conv5

  // transform.structured.generalize %conv3
  // TODO: needs drop unit dim as transform

  // TODO: loop unroll must invalidate handles properly...
  // transform.loop.unroll %loops#6 { factor = 8 } : !pdl.operation
  // transform.loop.unroll %loops#5 { factor = 2 } : !pdl.operation


  transform.yield
}
