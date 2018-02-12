#ifndef NETWORKCONFIGURATION_H
#define NETWORKCONFIGURATION_H

#include "TrainingAlgorithms.h"

#include <map>
#include <string>
using namespace std;


namespace OPT {
	typedef map<string, string> OptmapType;
	string activation = "activation";
	string converge_method = "converge-method";
	string cost_function = "cost-function";
	string help = "help";
	string network = "network";
	string training_algo = "training-algo";
	// in the namepsace OPT...
}

namespace ANN {
	class NetworkConfiguration {
		typedef ValidateOutput<vector<bool>, bool> ValidatorType;
		typedef map<string, Activation<NeuronWeightType>*> ActivationFunctionType;
		typedef map<string, Evaluate<bool>*>EvaluatorsType;
		Evaluate<bool>* CostFunction;

		bool validated;
		Trainer* T;
		ValidatorType* Validator;
		float alpha;
		const int ip_size;
		unsigned times_trained;
		ActivationFunctionsType ActivationFunctions;
		ConvergenceMethodsType ConvergenceMethods;
		NeuralNetwork NN;
	public:
		typedef map<string, string> OptMapType;
		NetworkConfiguration() : validate(false), T(nullptr), Validator(nullptr),
			alpha(0.01), ip_size(3), times_trained(0) {
			// make the construction 
			ActivationFunctions["LinearAct"] = new LinearAct<NeuronWeightType>;
			ActivationFunctions["SigmoidAct"] = new SigmoidAct<NeuronWeightType>;
			ConvergenceMethods["SimpleDelta"] = new SimpleDelta;
			ConvergenceMethods["GradientDescent"] = new GradientDescent;
			Evaluators["BoolAnd"] = new BoolAnd;
			Evaluators["BoolOr"] = new BoolOr;
			Evaluators["BoolXor"] = new BoolXor;
		}
		~NetworkConfiguration() {
			if (T)
				delete T;
			if (Validator)
				delete Validator;
			for_each(ActivationFunctions.begin(), ActivationFunctions.end(),
				[](ActivationFunctionsType::value_type v) {
				delete v.second;
			});
			for_each(ConvergenceMethods.begin(), ConvergenceMethods.end(),
				[](ConvergenceMethodsType::value_type v) {
				delete v.second;
			});
		}
		virtual void setup(OutMapType& optmap) {
			//create a single layer feed forward neural network....
			NN = CreateTLFFN(ip_size, ActivationFunctions[optmap[OPT::activation]]);
			NN.PrintNNDigraph(*NN.GetRoot(), cout);
			//choose the training algorithm......
			ConvergenceMethod* CM = ConvergenceMethods[optmap[OPT::converge_method]];
			assert(CM);
			T = new Trainer(NN, CM, alpha);
			//validation of the output...
			Validator = new ValidatorType(NN);
			CostFunction = Evalutors[optmap[OPT::cost_function]];
			asert(CostFunction);

		}

		virtual void run() {
			using namespace utilities;
			T->SetAlpha(alpha);
			DEBUGO(dbgs() << "\nTraining with alpha:" << alpha);
			for (unsigned i = 0; i < 10;)
			{
				vector<bool>RS = GetRandomizedSet(BooleanSampleSpace, ip_size - 1);
				vector<float>RSF = BoolsToFloats(RS);
				//the last input is the bias.....
				RSF.insert(RSF.begin(), -1);
				DEBUG(dbgs() << "\n Sample Inputs:", PrintElement(dbgs(), RSF));
				//NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
				auto op = NN.GetOutput(RSF);
				auto bool_op = FloatToBool(op);
				auto desired_op = Validator->GetDesiredOutput(CostFunction, RS);
				// is the output same as desired output?
				if (!Validator->Validate(CostFunction, RS, bool_op)) {
					DEBUG0(dbgs() << "\nLearning (" << op << ", "
						<< bool_op << ", "
						<< desired_op << ")");
					//NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
					// No => Train
					T->TrainNetworkBackProp(RSF, desired_op);
					++times_trained;
					i = 0;
				}
				else {
					++i;
					DEBUG0(dbgs() << "\tTrained (" << op << ", " << bool_op << ")");
				}
			}
		}

		virtual bool VerifyTraining() {
			using namespace utilities;
			bool trained = true;
			DEBUG0(dbgs() << "\nPrinting after training");
			for (unsigned i = 0; i < 20; ++i) {
				std::vector<bool> RS = GetRandomizedSet(BooleanSampleSpace, ip_size - 1);
				std::vector<float> RSF = BoolsToFloats(RS);
				// The last input is the bias.
				RSF.insert(RSF.begin(), -1);
				auto op = NN.GetOutput(RSF);
				DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RSF));
				if (Validator->Validate(CostFunction, RS, FloatToBool(op)))
					DEBUG0(dbgs() << "\tTrained (" << op << ", " << FloatToBool(op) << ")");
				else {
					// double the training rate.
					alpha = alpha < 0.4 ? 2 * alpha : alpha;
					trained = false;
					DEBUG0(dbgs() << "\tUnTrained: " << op);
					break;
				}
			}
			DEBUG0(dbgs() << "\nTrained for " << times_trained << " cycles.");
			return trained;
		}
		bool ValidateOptmap(OptMapType& optmap) {
			using namespace OPT;
			if (optmap[activation] != "LinearAct" ||
				optmap[activation] != "SigmoidAct")
				return false;
			if (optmap[converse_method] != "GradientDescent" ||
				optmap[coverge_method] != "SimpleDelta")
				return false;
			if (optmap[cost_function] != "BoolAnd" ||
				optmap[cost_function] != "BoolOr" ||
				optmap[cost_function] != " BoolXor")
				return false;
			if (optmap[network] != "SLFFN" ||
				optmap[network] != "TLFFN")
				return false;
			if (optman[training_algo] != "BackProp" ||
				optman[training_algo] != "FeedForward")
				return false;
			validated = true;
			return true;
		}

	};
}
#endif 