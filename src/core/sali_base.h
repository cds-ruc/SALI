#ifndef __SALI_BASE_H__
#define __SALI_BASE_H__

#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace sali {

class model_param{
public:
    long double a;
    long double b;
    model_param(long double _a,long double _b):a(_a),b(_b){}
    model_param()=default;
};

/*template <class T>
class LinearModelInterface{
    virtual inline int predict(T key) const=0;
    virtual inline double predict_double(T key) const=0;
    virtual inline void clear()=0;
    virtual inline void train_two(long double mid1_key,long double mid2_key,long double mid1_target,long double mid2_target)=0;

};*/
// Linear regression model
template <class T>
class LinearModel
{
public:
    long double a = 0; // slope
    long double b = 0; // intercept

    LinearModel() = default;
    LinearModel(long double a, long double b) : a(a), b(b) {}
    explicit LinearModel(const LinearModel &other) : a(other.a), b(other.b) {}

    inline int predict(T key) const
    {
      return std::floor(a * static_cast<long double>(key) + b);
    }

    inline double predict_double(T key) const
    {
      return a * static_cast<long double>(key) + b;
    }
    inline void clear(){
      a=b=0;
    }
    inline void train_two(long double mid1_key,long double mid2_key,long double mid1_target,long double mid2_target){
      a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
      b = mid1_target - a * mid1_key;
    }
};

template <class T>
class TwoLinearModel
{
public:
    int type=0;
    long double mid =0;
    long double a1 = 0; // slope
    long double b1 = 0; // intercept
    long double a2 = 0; // slope
    long double b2 = 0; // intercept

    TwoLinearModel() = default;
    TwoLinearModel(T mid,long double a1, long double b1,long double a2, long double b2) :mid(mid), a1(a1), b1(b1),a2(a2), b2(b2) {}
    explicit TwoLinearModel(const TwoLinearModel &other) : mid(other.mid),a1(other.a1), b1(other.b1),a2(other.a2),b2(other.b2) {}

    inline int predict(T key) const
    {
      if(type==0){
        return std::floor(a1 * static_cast<long double>(key) + b1);
      }
      if(static_cast<long double>(key)<mid){
        return std::floor(a1 * static_cast<long double>(key) + b1);
      }
      return std::floor(a2 * static_cast<long double>(key) + b2);
    }

    inline double predict_double(T key) const
    {
      if(type==0){
        return a1 * static_cast<long double>(key) + b1;
      }
      if(static_cast<long double>(key)<mid) {
        return a1 * static_cast<long double>(key) + b1;
      }
      return a2 * static_cast<long double>(key) + b2;
    }
    inline void clear(){
      a1=b1=a2=b2=mid=0;
    }
    inline void train_two(long double mid1_key,long double mid2_key,long double mid1_target,long double mid2_target){
      a1 = (mid2_target - mid1_target) / (mid2_key - mid1_key);
      b1 = mid1_target - a1 * mid1_key;
      type=0;
      /*a1= a2= (mid2_target - mid1_target) / (mid2_key - mid1_key);
      b1= b2= mid1_target - a1 * mid1_key;
      mid=(mid1_key+mid2_key)/2;*/
    }
};

template <class T>
class MultiLinearModel
{
public:
    model_param top_param;
    std::array<model_param,8> params;
    std::array<int,8> segment_size;
    std::array<int,8> segment_offset;
    int segment_count=0;


    inline int predict(T key) const
    {
      return 0;
    }

    inline double predict_double(T key) const
    {
      return predict_pos(key);
    }
    inline int predict_pos(T key) {
      double v1=top_param.a * static_cast<long double>(key) + top_param.b;
      if(params.size()==0){
        print();
      }
      int seg_id=0;
      if (v1 > std::numeric_limits<int>::max() / 2) {
        seg_id=segment_count-1;
      }else if (v1 < 0) {
        seg_id=0;
      }else{
        seg_id=std::min(segment_count-1, static_cast<int>(v1));
      }
      double v2=params[seg_id].a * static_cast<long double>(key) + params[seg_id].b;

      if (v2 > std::numeric_limits<int>::max() / 2) {
        return segment_offset[seg_id]+segment_size[seg_id]-1;
      }
      if (v2 < 0) {
        return segment_offset[seg_id];
      }

      return segment_offset[seg_id]+std::min(segment_size[seg_id]-1, static_cast<int>(v2));
    }

    inline void clear(){
    }
    inline void train_two(long double mid1_key,long double mid2_key,long double mid1_target,long double mid2_target){
      /*params.clear();
      segment_offset.clear();
      segment_size.clear();*/
      top_param.a=0;
      top_param.b=0;
      long double a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
      long double b = mid1_target - a * mid1_key;
      /*std::cout<<std::to_string(mid1_target)<<" .."<< std::to_string(mid2_target)<<std::endl;
      std::cout<<std::to_string(mid1_key)<<" .."<< std::to_string(mid2_key)<<std::endl;
      std::cout<<std::to_string(params.size())<<std::endl;
      if(v.size()>0){
        std::cout<<"+++++++++======================================================="<<std::endl;
      }
      std::cout<<a<<" "<< b<<std::endl;*/
      segment_count=1;

      //std::cout<<"address1 "<< &params[0]<<std::endl;
//      params.push_back(model_param(a,b));
//      segment_size.push_back(8);
//      segment_offset.push_back(0);
      params[0]=model_param(a,b);
      segment_size[0]=8;
      segment_offset[0]=0;
      //print();
      /*std::cout<<"+++++++++"<<std::endl;
      std::cout<<std::to_string(a)<<" "<< std::to_string(b)<<std::endl;
      std::cout<<"+++++++++"<<std::endl;*/

    }
    inline void print(){
      std::cout<<"==========="<<std::endl;
      std::cout<<top_param.a<<" "<< top_param.b<<std::endl;
      std::cout<<segment_count<<std::endl;
      for(int i=0;i<segment_count;i++){
        std::cout<<std::to_string(params[i].a)<<" "<< std::to_string(params[i].b)<<std::endl;
        std::cout<<segment_size[i]<<std::endl;
        std::cout<<segment_offset[i]<<std::endl;
      }
      std::cout<<"==========="<<std::endl;
    }
};

}

#endif
