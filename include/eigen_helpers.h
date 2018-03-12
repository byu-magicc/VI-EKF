#pragma once

#include <ros/ros.h>
#include <Eigen/Core>


template <class Derived>
bool vectorToMatrix(Eigen::MatrixBase<Derived>& mat, std::vector<double> vec)
{
  ROS_ASSERT(vec.size() == mat.rows()*mat.cols());
  if(vec.size() != mat.rows()*mat.cols())
    return false;
  for(unsigned i=0; i < mat.rows(); i++)
  {
    for(unsigned j=0; j < mat.cols(); j++)
    {
      mat(i,j) = vec[mat.cols()*i+j];
    }
  }
  return true;
}

template <class Derived>
void importMatrixFromParamServer(ros::NodeHandle nh, Eigen::MatrixBase<Derived>& mat, std::string param)
{
  std::vector<double> vec;
  if(!nh.getParam(param, vec))
  {
    ROS_FATAL("Could not find %s/%s on server. Zeros!",nh.getNamespace().c_str(),param.c_str());
    mat.setZero();
    return;
  }
  ROS_ERROR_COND(!vectorToMatrix(mat,vec),"Param %s/%s is the wrong size" ,nh.getNamespace().c_str(),param.c_str());
}
