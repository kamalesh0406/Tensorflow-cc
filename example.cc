#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main(){
	using namespace tensorflow; //Refers to the tensorflow namesapce for all the different data types.
	using namespace tensorflow::ops;
	Scope root = Scope::NewRootScope(); //This creates a new session and initializes the graphs.
	//Matrix A =  [3  2]
	//            [-1 0]
	auto A = Const(root, { { 3.f, 2.f}, {-1.f, 0.f}});
	//Vector b = [3 5]
	auto b = Const(root, { { 3.f, 5.f}});
	//Now we do the matrix multiplication of Ab^T
	auto v = MatMul(root.WithOpName('v'), A, b, MatMul::TransposeB(true))
	//Now let us look at the session

	std::vector<Tensor> outputs;
	//Initialize the session 
	CLientSession session(root);
	//Run and get the output of v
	TF_CHECK_OK(session.run({v}), &outputs);
	//The result should be outputs[0] = [19; -3]
	LOG(INFO) << outputs[0].matrix<float>();
	return 0;
}