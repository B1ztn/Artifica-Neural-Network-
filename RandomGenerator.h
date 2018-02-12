/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   RandomGenerator.h
 * Author: b1
 *
 * Created on February 8, 2018, 12:59 PM
 */

#ifndef RANDOMGENERATOR_H
#define RANDOMGENERATOR_H

#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

namespace utilities{
    typedef float RNType;
    vector<bool> BooleanSampleSpace{0,1};
    
    class RNG{
        double Min, Max;
        random_device rd;
        mt19937 e2;
        uniform_real_distribution<> dist;
    public:
        RNG(double min, double max)
        : Min(min), Max(max), e2(rd()), dist(min,max){}
        
        
        RNType Get()
        {
            return dist(e2);
        }
        
        unsigned GetLowerBound(){
            return floor(dist(e2));
        }
        
        unsigned GetUpperBound(){
            return ceil(dist(e2));
        }
        
        RNType GetBoolean(){
            return dist(e2) >= (Max-Min)/2.0 ? 1:0;
        }       
    };
    template<typename T>
    vector<T> GetRandomizedSet(const vector<T>& SampleSpace, 
            unsigned Size){
        RNG rng(0, SampleSpace.size());
        vector<T> RandomizedSet;
        for(size_t sz = 0; sz<Size; ++sz)
        {
            RandomizedSet.push_back(SampleSpace[rng.GetBoolean()]);
        }
        return RandomizedSet;
    }
} // namepsace utilities...

#endif /* RANDOMGENERATOR_H */

