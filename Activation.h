/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Activation.h
 * Author: b1
 *
 * Created on February 8, 2018, 12:19 PM
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H



#include <Debug.h>
#include <cmath>
using namespace std;


namespace ANN{
    //Activation function designed in the form of CRTP........
    template<typename WeightType>
    class Activation{
        public:
            virtual WeightType Act(WeightType t)const = 0;
            
            virtual WeightType Deriv(WeightType t)const =0;
            ~Activation()
            { }
    };
    
    template<typename WeightType>
    class LinearAct : public Activation<WeightType>{ // LinearACt extends or inherite the Activation...
        public:
            WeightType Act(WeightType w)const{
                return w;
            }
            // derivative of linear activation function w.r.t weight is 1
            
            WeightType Deriv(WeightType t)const{
                return WeightType(1);
            }
        
    };
    
    template<typename WeightType>
    class SigmoidAct : public Activation<WeightType>{
        public:
            WeightType Act(WeightType w)const{
                DEBUG2(dbgs() << "\n Sigmoid function Input: " << w
                            << ", Output:" << tanh(w));
                return tanh(w);
            }
            // derivative of tanh activation function w.r.t weight is 1 - tan(w)^2...
            WeightType Deriv(WeightType t)const{
                auto tmp = tanh(t);
                return 1-tmp*tmp;
            }  
    };
} // namespace ANN....

#endif /* ACTIVATION_H */

