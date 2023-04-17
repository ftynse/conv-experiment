transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.bufferization.eliminate_empty_tensors %arg0
  // %empties = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!pdl.operation) -> !transform.op<"tensor.empty">
  // transform.bufferization.empty_tensor_to_alloc_tensor %empties : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
  %arg1 = transform.bufferization.one_shot_bufferize %arg0 {
    bufferize_function_boundaries = true,
    function_boundary_type_conversion = 1 : i32 } : (!pdl.operation) -> !pdl.operation

  %foralls = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %forall:4 = transform.split_handles %foralls in [4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  transform.loop.forall_to_for %forall#0 : (!pdl.operation) -> !pdl.operation

  // TODO: return all "scf.fors"
  %fors = transform.structured.match ops{["scf.for"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %for:5 = transform.split_handles %fors in [5] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  transform.loop.unroll %for#0 {factor = 8 } : !pdl.operation
  transform.loop.unroll %for#1 {factor = 3 } : !pdl.operation
  transform.loop.unroll %for#2 {factor = 3 } : !pdl.operation

  transform.loop.forall_to_for %forall#1 : (!pdl.operation) -> !pdl.operation
  transform.loop.forall_to_for %forall#2 : (!pdl.operation) -> !pdl.operation
  transform.loop.forall_to_for %forall#3 : (!pdl.operation) -> !pdl.operation

  %f2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %f3 = transform.vector.transfer_to_scf %f2 : (!pdl.operation) -> !pdl.operation
  %f4 = transform.vector.lower_contraction %f3 lowering_strategy = parallelarith : (!pdl.operation) -> !pdl.operation
  %f5 = transform.vector.lower_transfer %f4 max_transfer_rank = 1 : (!pdl.operation) -> !pdl.operation
  %f6 = transform.vector.lower_transpose %f5 lowering_strategy = eltwise : (!pdl.operation) -> !pdl.operation
  %f7 = transform.vector.lower_shape_cast %f6 : (!pdl.operation) -> !pdl.operation

  transform.yield
}
