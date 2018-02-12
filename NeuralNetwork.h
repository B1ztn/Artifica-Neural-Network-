/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   NeuralNetwork.h
 * Author: b1
 *
 * Created on February 8, 2018, 12:58 PM
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "RandomGenerator.h"
#include "Activation.h"
#include "TypeConversions.h"
#include "CostFunction.h"
#include "TrainingAlgorithm.h"
#include "Activation.h"
#include "CostFunction.h"
#include <Debug.h>
#include <RandomGenerator.h>
#include <Distance.h>

#include <list>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
using namespace std;
namespace ANN{
    typedef float WeightType;
    typedef WeightType DendronWeightType;
    typedef WeightType NeuronWeightType;
    typedef unsigned LevelType;
    typedef unsigned IdType;
    typedef vector<WeightType> InputVector;
    struct NeuronType;
    
    template<typename WeightType>
    inline bool Valid(WeightType& w){
        return true;
    }
    
    inline bool IsRootNeuron(IdType Id)
    {
        return Id == 0;
    }
    
    inline bool IsOutNeuron(IdType Id){
        return Id == 1;
    }
    
    inline bool IsBiasNeuron(IdType Id){
        return Id == 2;
    }
    IdType GetNewId(){
        static IdType ID = 0;
        return ID++;
    }
    
    enum Direction{
        InvalidDirection = 0,
        Inwards,
        Outwards
    };
    
    utilities::FloatingPointDistance DendronDist;
    utilities::FloatingPointDistance NeuronDist;
    /// each one dendron ,,,, each dendron must be 
    // connected to two neurons.
    // template<typename DendronWeightType>
    
    struct DendronType{
        IdType Id;
        NeuronType* In;
        NeuronType* Out;
        DendronWeightType W;
        // delta of previous weight update
        DendronWeightType dW;
        // in.... dendron .....-> out .....it is a layer....
        // output of 'in' neuron feeds to the input of 'out' neuron...
        DendronType(IdType i, NeuronType* in, NeuronType* out, 
                    DendronWeightType w)
                : Id(i), In(in), Out(out), W(w), dW(DendronWeightType(0)){
            assert(In && Out);
        }
        
        void SetWeight(DendronWeightType w){
            assert(Valid(w));
            W=w;
        }
        // this function should be used to set negative weight
        // of bias neurons.......
        
        void SetBiasWeight(DendronWeightType w);
        bool operator == (const DendronType& d)const{
            return (In == d.In) && (Out= d.Out) &&d
            DendronDist.CloseEnough(d.W,W);
        }
        
        bool operator != (const DendronType& d)const{
            return !(*this == d);
        }
    };
    
    typedef list<DendronType*> DendronsRefListType;
    typedef set<DendronType*> DendronsRefSetType;
    typedef vector<DendronType*> DendronsRefVecType;
    // Represents one neuron. Each neuron may have multiple
    // incoming/outgoing connections.
    //template<typename NeuronWeightType>
    //
	//we try to create a  web networks.....with weights and nodes....

    struct NeuronType {
        typedef DendronType DendronType_t;
        IdType Id;
        IdType LayerNum;
        NeuronWeightType W; 
        NeuronWeightType dW;
        NeuronWeightType Output;
        bool EvalSigCalled;
        DendronsRefVecType Ins;
        DendronsRefVecType Outs;
        
        NeuronType(IdType i, unsigned ln, NeuronWeightType w)
                : Id(i), LayerNum(ln), W(w),
                dW(NeuronWeightType(0)), EvalSigCalled(false)
        { }

		NeuronType(IdType i, unsigned ln, DendronType* in,
			DendronType* out, NeuronWeightType w)
			: Id(i), LayerNum(ln), W(w),
			dW(NeuronWeightType(0)), EvalSigCalled(false) {
			Ins.push_back(in);
			Outs.push_back(out);
		}

		//define a new operator....
		bool operator==(const NeuronType n)const {
			return (n.LayerNum == LayerNum) && (n.Ins == Ins)
				&& (n.Outs == Outs) && NeuronDist.CloseEnough(n.W, W);
		}
		bool operator!=(const NeuronType n)const {
			return !(*this == n);
		}

		// in --------> (this Neuron)
		// (this Neuron) -------> out.....
		// Direction is w.r.t. this neuron.....
		// the direction is a new set for this model......

		void Connect(DendronType* d, Direction dir)
		{
			assert(d);
			assert(dir != Direction::InvalidDirection);
			//Self feedback is not supported yet.....
			assert(find(Ins.begin(), Ins.end(), d) == Ins.end());
			assert(find(Outs.begin(), Outs.end(), d) == Outs.end());
			if (dir == Direction::Inwards) {
				assert(d->Out == this);
				Ins.push_back(d);
			}
			else { //outwards .....
				assert(d->In == this);
				Outs.push_back(d);
			}
		}
		// layers... --->in--->(this Neuron)--->out--->
		// this Neuron (In) = in
		// this Neuron (Out) = out
		//
        
		void Connect(DendronType* in, DendronType* out)
		{
			Connect(in, Direction::Inwards);
			Connect(out, Direction::Outwards);

		}
        // we will set the weight for the node...
		void SetWeight(NeuronWeightType w)
		{
			assert(Valid(w));
			EvalSigCalled = true;
			W = w;
		}
		// remove the entry of dendron \d from the connections.....
		void Disconnect(DendronType* d, Direction dir)
		{
			assert(dir != Direction::InvalidDirection);
			DendronsRefVecType* ToBe = &Ins;
			if (dir == Direction::Outwards)
				ToBe = &Outs;
			DendronsRefVecType::iterator it = find(ToBe->begin(), ToBe->end(), d);
			assert(it != ToBe->end());
			ToBe->erase(it);
		}

		void Disconnect(DendronType* d)
		{
			Disconnect(d, Direction::Inwards);
			Disconnect(d, Direction::Outwards);

		}

		//this neuron -----n
		bool IsConnectedForward(NeuronType* n)
		{
			for (auto it = Outs.begin(); it != Outs.end(); ++it)
			{
				if ((*it)->Out == n)
					return true;
			}
			return false;
		}

		bool IsConnectedBackward(NeuronType* n) {
			for (auto it = Ins.begin(); it != Ins.end(); ++it)
			{
				if ((*it)->In == n)
					return true;
				return false;
			}
		}

		Direction IsConnected(NeuronType* n)
		{
			if (IsConnectedForward(n))
				return Direction::Outwards;
			if (IsConnectedBackward(n))
				return Direction::Inwards;
			return Direction::InvalidDirection;
		}

		Direction IsConnected(DendronType* d)
		{
			DendronsRefVecType::iterator it = find(Ins.begin(), Ins.end(), d);
			if (it != Ins.end())
				return Direction::Inwards;
			it = find(Outs.begin(), Outs.end(), d);
			if (it != Outs.end())
				return Direction::Outwards;
			return Direction::InvalidDirection;
		}
		//calcualte the sum of the output...
		DendronWeightType EvalSig() {
			assert(!IsRootNeuron(Id) && "Cannot use this function for root neurons");
			auto d_it = Ins.begin();
			NeuronWeightType Sum(0);
			while (d_it != Ins.end()) {
				DendronType* d = *d_it;
				//sum(weight of dendron * input).....
				Sum += (d->W)*(d->In->Output);
				++d_it;
			}
			W = Sum;
			EvalSigCalled = true;
			return Sum;
		}
		
		// To be used by the inner neurons only
		// and only after EvalSig has been called on this neuron.
		DendronWeightType EvalOp(const Activation<NeuronWeightType>& act) {
			assert(!IsRootNeuron(Id) && "Cannot use this function for root neurons");
			assert(EvalSigCalled && "First calculate the signal/ip on this neuron");
			Output = act.Act(W);
			DEBUG1(dbgs() << "\nEval n" << Id
				<< "(W:" << W << ", Output:" << Output << ")");
			EvalSigCalled = false;
			return Output;
		}

		ostream& operator<<(ostream os) {
			Print(os);
			return os;
		}

		template<typename Stream>
		void Print(Stream& s){
			s << "\n\tn" << Id
			<< " [ label = \"n" << Id << "(W:" << W << ")\"];";
                }
        };

	typedef list<NeuronType> NeuronsType;
	typedef list<DendronType> DendronsType;
	typedef list<NeuronType*> NeuronsRefListType;
	typedef vector<NeuronType*> NeuronsRefVecType;
	typedef set<NeuronType*> NeuronsRefSetType;
	/**
	*  ,--In--Cen---  -----
	* N --In--Cen---  ----- Out
	*  `--In--Cen---  -----
	* N is the root-neuron (kind of placeholder) and all the pseudo-edges
	* coming out of N are Input Dendrons.
	* This makes it easy to evaluate the network uniformly.
	* The neurons are the Adders of the weight*input of input dendrons.
	* @note As of now, the bias neuron is the first input neuron.
	* @note level 1 neurons are meant only for forwarding the input.
	*/

	class NeuralNetwork {
		NeuronsType Neurons;
		DendronsType Dendrons;
		typedef std::vector<std::vector<NeuronType*> > NeuronsByLayerType;
		typedef Activation<NeuronWeightType> ActivationFnType;
		const ActivationFnType* ActivationFn;
		NeuronsByLayerType NeuronsByLayer;

		//root is the first entry in Neurons....
		NeuronsType* RootNeuron;
		NeuronsType* OutNeuron;
		unsigned NumLayers;
	public:
		NeuralNetwork() : ActivationFn(nullptr), RootNeuron(nullptr), NumLayers(0) { }
		NeuralNetwork(const ActivationFnType* act) : ActivationFn(act), RootNeuron(nullptr), NumLayers(0) 
		{ }
		//initializes the network with just one neuron....
		NeuralNetwork(NeuronType& n, const ActivationFnType* act)
			: ActivationFn(act) {
			assert(act);
			Neurons.push_back(n);
			RootNeuron = &*Neurons.begin();
			NumLayers = 0;
		}
		// Root (the placeholder) has unit weight
		// so that evaluating subsequent stages becomes regular.

		NeuronsType::iterator CreateRoot() {
			assert(RootNeuron == NULL);
			// root is in first layer and has a weight 1...
			CreateNeuron(0, 1);
			RootNeuron = &*Neurons.begin();
			return Neurons.begin();
		}
		/*/ If the network size is known beforehand.
		void Resize(unsigned numNeurons, unsigned numDendrons) {
		Neurons.resize(numNeurons);
		Dendrons.resize(numDendrons);
		}*/

		void SetOutputNeuron(NeuronType& n)
		{
			OutNeuron = &n;
		}
		NeuronType& GetOutputNeuron() {
			assert(OutNeuron && "OutNeuron has not been initialized yet");
			return *OutNeuron;
		}
		//root is the first entry....
		NeuronsType::iterator GetRoot() {
			assert(RootNeuron);
			return Neurons.begin();
		}
		const ActivationFnType& GetActivationFunction() const {
			return *ActivationFn;
		}


		NeuronsType::iterator AddNeuron(NeuronType& n) {
			Neurons.push_back(n);
			return --Neurons.end();
		}


		// Use the iterator, don't reuse it.
		// It might have been invalidated.
		/// @param ln = Layer number where this neuron will go.

		NeuronsType::iterator CreateNeuron(unsigned ln, NeuronWeightType w = NeuronWeightType(0)){
			NeuronsType n(GetNewId(), ln, w);
			if (NumLayers < ln)
				NumLayers = ln;
			return AddNeuron(n);

		}
		// create a dendron that connects
		// i1 ------> i2
		// Use the iterator, don't reuse it.
		// It might have been invalidated.

		DendronsType::iterator CreateDendron(NeuronsType::iterator i1,
			NeuronsType::iterator i2, DendronWeightType w = DendronWeightType(0)) {
			DendronType d(GetNewId(), &*i1, &*i2, w);
			return AddDendron(d);
		}
		//dendron is another word "tree" in  Greeks....
		/*
			we try to add the nodes in tree...
			connect the nodes to different nodes....
		*/
		NeuronType::iterator AddDendron(DendronType& d)
		{
			Dendrons.push_back(d);
			return --Dendrons.end();
		}
		void Connect(NeuronType& n, DendronType& d, Direction direction) {
			if (direction == Inwards)
				n.Ins.push_back(&d);
			else {
				n.Outs.push_back(&d);
			}
		}

		void Connect(NeuronType& n, DendronType& d1, DendronType& d2)
		{
			n.Connect(&d1, &d2);
		}

		// Create a new dendron with n1 as output, n2 as input.
		// n1 ---------> n2

		void Connect(const NeuronType n1, const NeuronType n2,
			DendronWeightType w) {
			auto it1 = find(Neurons.begin(), Neurons.end(), n1);
			auto it2 = find(Neurons.begin(), Neurons.end(), n2);
			Connect(it1, it2, w);
		}
		// Create a new dendron with i1 as output, i2 as input.
		// i1 ---------> i2
		// Enter the connection in each neuron.

		void Connect(NeuronsType::iterator i1, NeuronsType::iterator i2,
			DendronWeightType w) {
			assert(i1 != Neurons.end());
			assert(i2 != Neurons.end());
			assert(i1 != i2);
			assert(!i1->IsConnected(&*i2));
			//Any connection from root has to be a unit weight....
			if (&*i1 == RootNeuron)
				assert(w == DendronWeightType(1));
			auto dp = CreateDendron(i1, i2, w);
			// d is the output of i1......
			i1->Connect(&*dp, Direction::Outwards);
			// d is the input of the i2.....
			i2->Connect(&*dp, Direction::Inwards);
		}
		void Remove(NeuronType& n)
		{
			assert(n != *RootNeuron);
			NeuronsType::iterator it = find(Neuron.begin(), Neurons.end(), n);
			assert(it != Neurons.end());
			for (auto ins = n.Ins.begin(); ins != n.Ins.end(); ++ins)
			{
				Remove(**ins);
			}
			for (auto outs = n.Outs.begin(); outs != n.Outs.end(); ++outs)
			{
				Remove(**outs);
			}
			Neurons.erase(it);
		}

		void Remove(DendronType& d) {
			DendronsType::iterator it = find(Dendrons.begin(), Dendrons.end(), d);
			assert(it != Dendrons.end());
			// d.IN -> out -In ----> dendron -->Out-In ----->d.Out
			d.In->Disconnect(&d, Direction::Outwards);
			d.Out->Disconnect(&d, Direction::Inwards);
			Dendrons.erase(it);

		}
		NeuronWeightType GetOutput(const vector<DendronWeightType> Ins) {
			assert(RootNeuron->Outs.size() == Ins.size());
			assert(ActivationFn && "Uninitalized activation function");
			auto ip = Ins.begin();
			NeuronsRefVecType NeuronRefs;
			// Root. First propagate the input multiplied by
			// input dendron weights to the layer 1 neurons.
			NeuronType* Current = RootNeuron;
			NeuronsRefSetType InNeuronRefs1;
			auto it = Current->Outs.begin();
			for (; it != Current->Outs.end(); ++it)
			{
				DendronType* d = *it;
				d->Out->SetWeight((*ip)*d->W);
				DEBUG1(dbgs() << "\nWt: " << d->W << ", ip:" << *ip;
				d->Out->Print(dbgs()););
				d->Out->EvalOp(*ActivationFn);
				auto dit = d->Out->Outs.begin();
				for (; dit != d->Out->Outs.end(); ++dit)
				{
					InNeuronRefs1.insert((*dit)->Out);
				}
				++ip;
			}
			/// @note: The output neuron might be evaluated multiple times.
			/// but that's okay for now because it keeps the algorithm simple.
			for (auto it = InNeuronRefs1.begin(); it != InNeuronRefs1.end(); ++it)
				NeuronRefs.push_back(*it);
			InNeuronRefs1.clear();
			NeuronWeightType op;
			while (!NeuronRefs.empty()) {
				/// @todo: Optimize this. Get a pointer to the neuron being
				// inserted to and check for size() > 1 using the pointer
				// to the set. That way I can avoid a copy.
				DEBUG1(dbgs() << "\nPrinting the neurons inserted:";
				std::for_each(NeuronRefs.begin(), NeuronRefs.end(),
					[&](NeuronType* np) { np->Print(dbgs()); dbgs() << " "; });
				);

				for_each(NeuronRefs.begin(), NeuronRefs.end(), [&](NeuronType* N) {
					N->EvalSig();
					op = N->EvalOp(*ActivationFn);
					for_each(N->Outs.begin(), NeuronRefs.end(), [&](DendronType* din) {InNeuronRefs1.insert(din->Out); });
				});
				NeuronRefs.clear();
				for (auto it = InNeuronRefs1.begin(); it != InNeuronRefs1.end(); ++it)
				{
					NeuronRefs.push_back(*it);
				}
                                InNeuronRefs1.clear();
                        }
				return op;
			}
                        
                        unsigned GetTotalLayers() const {
                          return NumLayers;
                        }
                        
			const NeuronsByLayerType& GetNeuronByLayer()const {
				return NeuronsByLayer;
			}
			void ClearNeuronsByLayer() {
				NeuronsByLayer.clear();
			}
			//generate the layer with neurons....

			NeuronsByLayerType GenNeuronsByLayer() {
				assert(NeuronsByLayer.empty());
				NeuronsByLayer.resize(GetTotalLayers() + 1);
				for (auto it = Neurons.begin(); it != Neurons.end(); ++it)
				{
					NeuronType& n = *it;
					NeuronsByLayer[n.LayerNum].push_back(&n);
					DEBUG0(dbgs() << "\nNeuron#" << n.Id << ", Layer" << n.LayerNum);

				}
				return NeuronsByLayer;
			}
			template<typename Stream>
			void PrintNeurons(Stream& s)
			{
				for (auto ni = Neurons.begin(); ni != Neurons.end(); ++ni)
				{
					ni->Print(s);
				}
			}

			template<typename Stream>
			void PrintConnectionsDFS(NeuronType& root, DendronsRefSetType& Printed,
				Stream& s) {
				for (auto dp = root.Outs.begin(); dp != root.Outs.end();dp++) {
					if (Printed.find(*dp) == Printed.end())
						PrintDendron(s, **dp);
					Printed.insert(*dp);
					PrintConnectionsDFS(*(*dp)->Out, Printed, s);
				}
			}
			// Use a BFS method to print the neural network.
			template<typename Stream>
			void PrintNNDigraph(NeuronType& root, Stream& s,
				const std::string& Name = "") {
				DendronsRefSetType Printed;
				s << "\ndigraph " << Name << " {\n";
				PrintNeurons(s);
				PrintConnectionsDFS(root, Printed, s);
				s << "\n}\n";
			}
		};

		// Single Layer Feed Forward network.
		// ,----IN ----
		// N----IN ---- Out
		// `----IN ----
		// @param NumNeurons = Number of input layer neurons.
		NeuralNetwork CreateSLFFN(unsigned NumNeurons,
			const Activation<NeuronWeightType>* act) {
			using namespace utilities;
			RNG rnd(-1, 1);
			NeuralNetwork nn(act);
			auto root = nn.CreateRoot();
			auto out = nn.CreateNeuron(2, 0);
			auto bias = nn.CreateNeuron(1, 0);
			nn.SetOutputNeuron(*out);
			nn.Connect(root, bias, 1.0);
			nn.Connect(bias, out, 1.0);
			for (unsigned i = 0; i < NumNeurons - 1; ++i) {
				auto in = nn.CreateNeuron(1, 0);
				nn.Connect(root, in, 1.0);
				nn.Connect(in, out, rnd.Get());
			}
			return nn;
		}
		// Two Layer Feed Forward network
		// ,---- IN ---- IN
		// ,---- IN ---- IN
		// N-----IN ---- IN-- Out
		// `---- IN ---- IN//
		// `---- IN ---- IN/
		NeuralNetwork CreateTLFFN(unsigned NumNeurons,
			const Activation<NeuronWeightType>* act) {
			using namespace utilities;
			RNG rnd(-1, 1);
			NeuralNetwork nn(act);
			std::vector<NeuronsType::iterator> L2Neurons;
			auto root = nn.CreateRoot();
			auto out = nn.CreateNeuron(3, 0);
			nn.SetOutputNeuron(*out);
			auto bias = nn.CreateNeuron(1, 0);
			// Any connection from root has to be a unit weight.
			nn.Connect(root, bias, 1.0);
			nn.Connect(bias, out, 1.0);
			// Connect bias neurons to each l2 neurons,
			// and each l2 neurons to out.
			for (unsigned i = 0; i < NumNeurons - 1; ++i) {
				auto l2 = nn.CreateNeuron(2, 0);
				nn.Connect(bias, l2, 1.0);
				nn.Connect(l2, out, rnd.Get());
				L2Neurons.push_back(l2);
			}
			// For each layer1 neuron make connection with it
			// to all the layer 2 neurons.
			for (unsigned i = 0; i < NumNeurons - 1; ++i) {
				auto l1 = nn.CreateNeuron(1, 0);
				nn.Connect(root, l1, 1.0);
				for (unsigned j = 0; j < NumNeurons - 1; ++j) {
					nn.Connect(l1, L2Neurons[j], rnd.Get());
				}
			}
			return nn;
		}

		template<typename Inputs, typename Output>
		class ValidateOutput {
		private:
			NeuralNetwork& NN;
		public:
			ValidateOutput(NeuralNetwork& nn)
				:NN(nn)
			{ }

			template<typename T>
			Output GetDesiredOutput(Evaluate<T>* eval, Inputs Ins) const {
				//return std::accumulate(Ins.begin(), Ins.end(), Evaluate<T>::init_value,
				//                       eval);
				typename Evaluate<T>::init_type init = eval->init_value;
				DEBUG2(dbgs() << "\tInit: " << init);
				std::for_each(Ins.begin(), Ins.end(),
					[&init, &eval](typename Inputs::value_type val) {
					init = eval->operator()(init, val);
				});
				DEBUG2(dbgs() << ", Desired output: " << init);
				return init;
			}

			template<typename T>
			bool Validate(Evaluate<T>* eval, Inputs Ins, Output op) const {
				return GetDesiredOutput(eval, Ins) == op;
			}
		};
 // namespace ANN
        }
#endif // ANN_NEURAL_NETWORK_H
#endif /* NEURALNETWORK_H */

