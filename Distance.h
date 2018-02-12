/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Distance.h
 * Author: b1
 *
 * Created on February 8, 2018, 1:44 PM
 */

#ifndef DISTANCE_H
#define DISTANCE_H

#include <cmath>
namespace utilities{
static constexpr float FPDelta = 0.00001;
template<typename DT>

struct Distance{
    template<typename T>
    T operator()(T t1, T t2)const {
        const DT& dt = static_cast<const DT>(*this);
        return dt.CloseEnough(t1, t2);
    }
    
    template<typename T>
    bool CloseEnough(T t1, T t2)const { 
        DT& dt = static_cast<const DT&>(*this);
        return dt.CloseEnough(t1,t2);
    }
};


struct IntegralDistance : public Distance<IntegralDistance>{
    int operator()(int t1, int t2)const {
        return t1   - t2;
    }
    bool CloseEnough(int t1, int t2)const {
        return t1 == t2;
    }
};

struct FloatingPointDistance : public Distance<IntegralDistance>{
    int operator()(float t1, float t2)const{
        return t1 - t2;
    }
    
    bool CloseEnough(float t1, float t2)const{
        return fabs(t1-t2)<FPDelta;
    }
};

}
#endif /* DISTANCE_H */

