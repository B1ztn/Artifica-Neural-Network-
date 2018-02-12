/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   TrainingAlgorithm.h
 * Author: b1
 *
 * Created on February 12, 2018, 10:53 AM
 */

#ifndef ANN_TRAINING_ALGORITHMS_H
#define ANN_TRAINING_ALGORITHMS_H

#include "NeuralNetwork.h"
#include "RandomGenerator.h"
#include "Activation.h"

#include <cassert>
namespace ANN {
	//Trainning algorithm only calculate the deltas.....
	// we need to set the hiddenlayers and outputlayers....

	struct ConvergenceMethod {
		virtual void OutputNode(NeuronType& d, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act)const = 0;
		virtual void HiddenNode(NeuronType& n, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act)const = 0;

		virtual ~ConvergenceMethod()
		{}
	};


	struct GradientDescent : public ConvergenceMethod {
		void OutputNode(NeuronType& n, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act) const {
			auto error = (n.Output - desired_op);
			// mean square error
			/// @todo: Optimize when activation is tanh, in that case
			/// n.dW = 2*error*(1-n.Output*n.Output);
			n.dW = 2 * error*act.Deriv(n.W);
		}
		void HiddenNode(NeuronType& n, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act)const {
			//n.dW = deriv(output)*Sum(delta(output-neuron))
			n.dW = NeuronWeightType(0);
			for (auto it = n.Outs.begin(); it != n.Outs.end(); ++it)
			{
				n.dW += ((*it)->W)*((*it)->Out->dW);
			}
			/// @todo: Optimize when activation is tanh, in that case
			/// n.dW = n.dW*(1-n.Output*n.Output);
			n.dW = (N.dW)*(act.Deriv(n.W));

		}
	};
	struct SimpleDelta : public ConvergenceMethod {
		void OutputNode(NeuronType& n, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act)const {
			auto error = (n.Output - desired_op);
			// mean square error
			/// @todo: Optimize when activation is tanh, in that case
			/// n.dW = 2*error*(1-n.Output*n.Output);
			n.dW = 2 * error*act.Deriv(n.W);
		}
		void HiddenNode(NeuronType& n, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act)const {
			auto error = (n.Output - desired_op);
			// mean square error
			/// @todo: Optimize when activation is tanh, in that case
			/// n.dW = 2*error*(1-n.Output*n.Output);

			n.dW = 2 * error*act.Deriv(n.W);
		}
		void HiddenNode(NeuronType& n, NeuronWeightType desired_op,
			const Activation<NeuronWeightType>& act)const {
			assert(0 && "No hidden node training for feed forward network");
		}
	};

	// Trains the neural network when the training algorithm
	// is provided. Each dendron is trained by the breadth first
	// traversal of the network starting from root node.
	// Assumptions: The dendrons have the weight and neurons
	// are the summers.
        // actually we can change the conergence methods....
        
        
        
	class Trainer {
		NeuralNetwork& NN;
		ConvergenceMethod* CM;
		float alpha;
		utilities::RNG noise;
	public:
		Trainer(NeuronNetwork& nn, ConvergenceMethod* cm, DendronWeightType al)
			: NN(nn), CM(cm), alpha(al), noise(-alpha/5.0, alpha/5.0)
		{ }
		const ConvergenceMethod& GetTrainingAlgorithm() {
			return *CM;
		}
		const NeuronNetwork& GetNeuralNetwork()const {
			return NN;
		}
		void SetAlpha(DendronWeightType al) {
			alpha = al;
		}
		void TrainOutputNeuron(NeuronType& n)
		{
			CM->HiddenNode(n, NeuronWeightType(0), NN.GetActivationFunction());
		}

		void TrainHiddenNeuron(NeuronType& n)
		{
			// The second parameter (Desired output), is not used currently.
			CM->HiddenNode(n, NeuronWeightType(0), NNN.GetActivationFunction());
		}

		void TrainDendron(DendronType& dp) {
			// weight updated = alpha*input*delta....
			// dp->dW += dp->dW+delta;//for momentum....
			DEBUG0(dbgs() << "d.W=" << dp.W
				<< ", x.Out=" << dp.In->Output
				<< ", n.dW=" << dp.Out->dW);
			//dp.dW = alpha*output(input neuron)*delta(output neuron)
			dp.dW = -alpha*((dp.In->Output)*(dp.Out->dW));
			dp.W += dp.dW;
			DEBUG0(dbgs() << ", d.W'=" << dp.W << ", d.dW=" << dp.dW);
		}

		// In feedforward network delta for all the dendrons are the same,
		// and equal to the delta of output neuron.
		void TrainDendronFF(DendronType& dp, NeuronWeightType delta)
		{
			// weight update = alpha*input*delta
			//dp->dW += dp->dW + delta; // for momentum.
			DEBUG0(dbgs() << "d.W=" << dp.W
				<< ", x.Out=" << dp.In->Output
				<< ", delta=" << delta);

			// dp.dW = alpha*output(input neuron)*delta(output neuron)
			dp.dW = -alpha*delta;
			dp.W += dp.dW;
			DEBUG0(dbgs() << ", d.W'=" << dp.W << ", d.dW=" << dp.dW);
		}

		/** Breadth First traversal of network == Feed forward training.
		*visited the neigbour first....
		* This is good for boolean functions like AND/OR
		* @todo: Separate the training of level-1 neurons
		* in a separate loop. And then train all the subsequent
		* neurons. That would improve performance in
		* multi-layer networks with a lot of inner layers.
		*/
		template<typename InputSet, typename OutputType>
		void TrainNetworkFeedForward(const InputSet& Ins, OutputType desired_op) {
			NeuronType& root = *NN.GetRoot();
			NeuronType& out = NN.GetOutputNeuron();
			assert(root.Outs.size() == Ins.size());
			TrainOutputNeuron(out, desired_op);
			NeuronsRefVecType ToBeTrained;
			NeuronsRefSetType ToBeTrained2;
			// level 1 neurons are meant only for forwarding the input.
			for (auto dp = root.Outs.begin(); dp != root.Outs.end(); ++dp)
			{
				NeuronType* np = (*dp)->Out;
				if (!IsBiasNeuron(np->Id)) {
					auto dp12 = np->Outs.begin();
					for (; dp12 != np->Outs.end(); ++dp12)
					{
						DEBUG0(dbgs() << "\nTraining d" << (*dpl2)->Id << "\t");
						TrainDendronFF(**dp12, out.dW);
						ToBeTrained2.insert((*dp12)->Out);
					}
				}
			}
			for (auto it = ToBeTrained2.begin(); it != ToBeTrained2.end(); ++it)
			{
				ToBeTrained.push_back(*it);
			}
			ToBeTrained2.clear();
			while (!ToBeTrained.empty())
			{
				auto nit = ToBeTrained.begin();
				for (; nit != ToBeTrained.end(); ++nit)
				{
					NeuronType* np = *nit;
					auto dp12 = np->Outs.begin();
					for (; dp12 != np->Outs.end(); ++dp12)
					{
						DEBUG0(dbgs() << "\nTraining d" << (*dpl2)->Id << "\t");
						TrainDendronFF(**dp12, out.dW);
						ToBeTrained2.insert((*dp12)->Out);
					}
				}
				ToBeTrained.claer();
				for (auto it = ToBeTrained2.begin(); it != ToBeTrained2.end(); ++it)
				{
					ToBeTrained.push_back(*it);
				}
				ToBeTrained2.clear();

			}

		}
		// Only works for scalar output, i.e only one output neuron....
		template<typename InputSet, typename OutputType>
		void TrainNetworkBackProp(const InputSet& Ins, OutputType desired_op)
		{
			NeuronType& root = *NN.GetRoot();
			NeuronType& out = NN.GetOutputNeuron();
			assert(root.Outs.size() == Ins.size());
			NeuronsRefVecType ToBeTrained;
			NeuronsRefSetType ToBeTrained2;
			TrainOutputNeuron(out, desired_op);
			for (auto di = out.Ins.begin(); di != out.Ins.end(); ++di)
			{
				if (!IsBiasNeuron((*di)->In->Id)) {
					ToBeTrained.push_back((*id)->In);
					DEBUG0(dbgs() << "\nTraining d" << (*di)->Id << "\t");
					TrainDendron(**di);
				}
			}
			// train all the neurons and dendrons until the input layer.
			// assuming the network is fully connected.
			while (!ToBeTrained.empty()) {
				for (auto np = ToBeTrained.begin(); np != ToBeTrained.end(); ++np)
				{
					NeuronType& n = **np;
					if (!IsRootNeuron(dp->In->Id) && !IsBiasNeuron(dp->In->Id))
					{
						DEBUG0(dbgs() << "\nTraining dendron:" << dp->Id << "\t");
						TrainDendron(*dp);
						ToBeTrained2.insert(dp->In);
					}
				}
			}
			ToBeTrained.clear();
			for (auto it = ToBeTrained2.begin(); it != ToBeTrained2.end(); ++it)
			{
				ToBeTrained.push_back(*it);
			}
			ToBeTrained2.clear();

		}

	};

} // namespace ANN...

#endif // ANN_TRAINING_ALGORITHMS_H

