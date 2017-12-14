#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include <iostream>

namespace quat {


class Quaternion
{


private:
  Eigen::Vector4d arr_;

public:

  Quaternion(Eigen::Vector4d arr) : arr_(arr) {}

  inline double w() const { return arr_(0); }
  inline double x() const { return arr_(1); }
  inline double y() const { return arr_(2); }
  inline double z() const { return arr_(3); }
  inline void setW(double w) { arr_(0) = w; }
  inline void setX(double x) { arr_(1) = x; }
  inline void setY(double y) { arr_(2) = y; }
  inline void setZ(double z) { arr_(3) = z; }
  inline Eigen::Vector4d elements() const { return arr_;}

  Quaternion operator* (const Quaternion q) { return otimes(q); }
  Quaternion& operator *= (const Quaternion q)
  {
    arr_ <<  w() * q.w() - x() *q.x() - y() * q.y() - z() * q.z(),
             w() * q.x() + x() *q.w() + y() * q.z() - z() * q.y(),
             w() * q.y() - x() *q.z() + y() * q.w() + z() * q.x(),
             w() * q.z() + x() *q.y() - y() * q.x() + z() * q.w();
  }

  Quaternion& operator= (const Quaternion q)
  {
    arr_[0] = q.w();
    arr_[1] = q.x();
    arr_[2] = q.y();
    arr_[3] = q.z();
  }

  Quaternion operator+ (const Eigen::Vector3d v) { return boxplus(v); }
  Quaternion& operator+= (const Eigen::Vector3d v)
  {
    double norm_v = v.norm();
    Eigen::Vector3d tmp = v;

    Eigen::Vector4d q_new;
    if (norm_v > 1e-4)
    {
        tmp *= std::sin(norm_v / 2.)/norm_v;
        q_new << std::cos(norm_v/2.0), v(0), v(1), v(2);

        arr_ <<  w() * q_new(0) - x() *q_new(1) - y() * q_new(2) - z() * q_new(3),
                 w() * q_new(1) + x() *q_new(0) + y() * q_new(3) - z() * q_new(2),
                 w() * q_new(2) - x() *q_new(3) + y() * q_new(0) + z() * q_new(1),
                 w() * q_new(3) + x() *q_new(2) - y() * q_new(1) + z() * q_new(0);
    }
    else
    {
      tmp *= 0.5;
      q_new << 0.0, v(0), v(1), v(2);
      q_new <<  w() * q_new(0) - x() *q_new(1) - y() * q_new(2) - z() * q_new(3),
                w() * q_new(1) + x() *q_new(0) + y() * q_new(3) - z() * q_new(2),
                w() * q_new(2) - x() *q_new(3) + y() * q_new(0) + z() * q_new(1),
                w() * q_new(3) + x() *q_new(2) - y() * q_new(1) + z() * q_new(0);
       arr_ += q_new;
       arr_ /= arr_.norm();
    }
  }

  Eigen::Vector3d operator- (const Quaternion q)
  {
    Quaternion dq = q.inverse().otimes(*this);
    if (dq.w() < 0.0)
    {
      dq.elements() *= -1.0;
    }
    return log(dq);
  }

  static Eigen::Matrix3d skew(const Eigen::Vector3d v)
  {
    Eigen::Matrix3d cum_sum;
    cum_sum << 0.0, -v(2), v(1),
        v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    return cum_sum;
  }

  static Quaternion exp(const Eigen::Vector3d v)
  {
    double norm_v = v.norm();

    Eigen::Vector4d q_arr;
    if (norm_v > 1e-4)
    {
      double v_scale = std::sin(norm_v/2.0)/norm_v;
      q_arr << std::cos(norm_v/2.0), v_scale*v(0), v_scale*v(1), v_scale*v(2);
    }
    else
    {
      q_arr << 1.0, v(0)/2.0, v(1)/2.0, v(2)/2.0;
      q_arr /= q_arr.norm();
    }
    return Quaternion(q_arr);
  }

  static Eigen::Vector3d log(const Quaternion q)
  {
    Eigen::Vector3d v = q.elements().block<3,1>(1, 0);
    double w = q.elements()(0,0);
    double norm_v = v.norm();

    Eigen::Vector3d out;
    if (norm_v < 1e-8)
    {
      out.setZero();
    }
    else
    {
      out = 2.0*std::atan2(norm_v, w)*v/norm_v;
    }
    return out;
  }

  static Quaternion from_R(const Eigen::Matrix3d m)
  {
    Eigen::Vector4d q;
    double tr = m.trace();

    if (tr > 0)
    {
      double S = std::sqrt(tr+1.0) * 2.;
      q << 0.25 * S,
           (m(1,2) - m(2,1)) / S,
           (m(2,0) - m(0,2)) / S,
           (m(0,1) - m(1,0)) / S;
    }
    else if ((m(0,0) > m(1,1)) && (m(0,0) > m(2,2)))
    {
      double S = std::sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2.;
      q << (m(1,2) - m(2,1)) / S,
           0.25 * S,
           (m(1,0) + m(0,1)) / S,
           (m(2,0) + m(0,2)) / S;
    }
    else if (m(1,1) > m(2,2))
    {
      double S = std::sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2.;
      q << (m(2,0) - m(0,2)) / S,
           (m(1,0) + m(0,1)) / S,
           0.25 * S,
           (m(2,1) + m(1,2)) / S;
    }
    else
    {
      double S = std::sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2.;
      q << (m(0,1) - m(1,0)) / S,
           (m(2,0) + m(0,2)) / S,
           (m(2,1) + m(1,2)) / S,
           0.25 * S;
    }
    return Quaternion(q);
  }

  static Quaternion from_axis_angle(const Eigen::Vector3d axis, const double angle)
  {
    double alpha_2 = angle/2.0;
    Eigen::Vector4d arr;
    arr << std::cos(alpha_2), axis(0)*alpha_2, axis(1)*alpha_2, axis(2)*alpha_2;
    arr /= arr.norm();
    return Quaternion(arr);
  }

  static Quaternion from_euler(const double roll, const double pitch, const double yaw)
  {
    double cp = std::cos(roll/2.0);
    double ct = std::cos(pitch/2.0);
    double cs = std::cos(yaw/2.0);
    double sp = std::sin(roll/2.0);
    double st = std::sin(pitch/2.0);
    double ss = std::sin(yaw/2.0);

    Eigen::Vector4d arr;
    arr << cp*ct*cs - sp*st*ss,
           sp*st*cs + cp*ct*ss,
           sp*ct*cs + cp*st*ss,
           cp*st*cs - sp*ct*ss;
    return Quaternion(arr);
  }

  static Quaternion from_two_unit_vectors(const Eigen::Vector3d u, const Eigen::Vector3d v)
  {
    Eigen::Vector4d q_arr;

    double d = u.dot(v);
    if (d < 1.0)
    {
      double invs = 1.0/std::sqrt((2.0*(1.0+d)));
      Eigen::Vector3d xyz = skew(u)*v*invs;
      q_arr(0) = 0.5/invs;
      q_arr.block<3,1>(1,0)=xyz;
      q_arr /= q_arr.norm();
    }
    else
    {
      q_arr << 1, 0, 0, 0;
    }
    return Quaternion(q_arr);
  }

  static Quaternion Identity()
  {
    Eigen::Vector4d q_arr;
    q_arr << 1.0, 0, 0, 0;
    return Quaternion(q_arr);
  }

  static Quaternion Random()
  {
    Eigen::Vector4d q_arr;
    q_arr.setRandom();
    q_arr /= q_arr.norm();
    return Quaternion(q_arr);
  }

  Eigen::Vector3d euler()
  {
    Eigen::Vector3d out;
    out << std::atan2(2.0*w()*x()+y()*z(), 1.0-2.0*(x()*x() + y()*y())),
        std::asin(2.0*(w()*y() - z()*z())),
        std::atan2(2.0*w()*z()+x()*y(), 1.0-2.0*(y()*y() + z()*z()));
    return out;
  }

  Eigen::Matrix3d R()
  {
    double wx = w()*x();
    double wy = w()*y();
    double wz = w()*z();
    double xx = x()*x();
    double xy = x()*y();
    double xz = x()*z();
    double yy = y()*y();
    double yz = y()*z();
    double zz = z()*z();
    Eigen::Matrix3d out;
    out << 1. - 2.*yy - 2.*zz, 2.*xy + 2.*wz, 2.*xz - 2.*wy,
        2.*xy - 2.*wz, 1. - 2.*xx - 2.*zz, 2.*yz + 2.*wx,
        2.*xz + 2.*wy, 2.*yz - 2.*wx, 1. - 2.*xx - 2.*yy;
    return out;
  }

  Quaternion copy()
  {
    Eigen::Vector4d tmp = arr_;
    return Quaternion(tmp);
  }

  void normalize()
  {
    arr_ /= arr_.norm();
  }

  Eigen::MatrixXd rot(Eigen::MatrixXd v)
  {
    Eigen::Matrix3d skew_xyz = skew(arr_.block<3,1>(1,0));
    Eigen::MatrixXd t = 2.0*skew_xyz * v;
    return v + w() * t + skew_xyz * t;
  }

  Eigen::Vector3d invrot(Eigen::Vector3d v)
  {
    Eigen::Matrix3d skew_xyz = skew(arr_.block<3,1>(1,0));
    Eigen::Vector3d t = 2.0*skew_xyz * v;
    return v - w() * t + skew_xyz * t;
  }

  Quaternion& inv()
  {
    arr_.block<3,1>(1,0) *= -1.0;
  }

  Quaternion inverse() const
  {
    Eigen::Vector4d tmp = arr_;
    tmp.block<3,1>(1,0) *= -1.0;
    return Quaternion(tmp);
  }

  Quaternion otimes(const Quaternion q)
  {
    Eigen::Vector4d new_arr;
    new_arr <<  w() * q.w() - x() *q.x() - y() * q.y() - z() * q.z(),
                w() * q.x() + x() *q.w() + y() * q.z() - z() * q.y(),
                w() * q.y() - x() *q.z() + y() * q.w() + z() * q.x(),
                w() * q.z() + x() *q.y() - y() * q.x() + z() * q.w();
    return Quaternion(new_arr);
  }

  Quaternion boxplus(Eigen::Vector3d delta)
  {
    return otimes(exp(delta));
  }
};

std::ostream& operator<< (std::ostream& os, const Quaternion& q)
{
  os << "[ " << q.w() << ", " << q.x() << "i, " << q.y() << "j, " << q.z() << "k]";
  return os;
}

}
