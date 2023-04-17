transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %f = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %f2 = transform.structured.vectorize %f
}
