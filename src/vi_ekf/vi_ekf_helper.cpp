#include "vi_ekf.h"

namespace vi_ekf
{

/**
 * @brief VIEKF::boxplus
 * @param x 
 * @param dx
 * @param out = x [+] dx
 */
void VIEKF::boxplus(const xVector& x, const dxVector& dx, xVector& out) const
{
  out.block<6,1>((int)xPOS, 0) = x.block<6,1>((int)xPOS, 0) + dx.block<6,1>((int)dxPOS, 0);
  out.block<4,1>((int)xATT, 0) = (Quat(x.block<4,1>((int)xATT, 0)) + dx.block<3,1>((int)dxATT, 0)).elements();
  out.block<7,1>((int)xB_A, 0) = x.block<7,1>((int)xB_A, 0) + dx.block<7,1>((int)dxB_A, 0);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(xZ+i*5,0) = q_feat_boxplus(Quat(x.block<4,1>(xZ+i*5,0)), dx.block<2,1>(dxZ+3*i,0)).elements();
    out(xZ+i*5+4) = x(xZ+i*5+4) + dx(dxZ+3*i+2);
  }
}

/**
 * @brief VIEKF::boxminus
 * @param x1
 * @param x2
 * @param out = x1 [-] x2
 */
void VIEKF::boxminus(const xVector &x1, const xVector &x2, dxVector &out) const
{
  out.block<6,1>((int)dxPOS, 0) = x1.block<6,1>((int)xPOS, 0) - x2.block<6,1>((int)xPOS, 0);
  out.block<3,1>((int)dxATT, 0) = (Quat(x1.block<4,1>((int)xATT, 0)) - Quat(x2.block<4,1>((int)xATT, 0)));
  out.block<7,1>((int)dxB_A, 0) = x1.block<7,1>((int)xB_A, 0) - x2.block<7,1>((int)xB_A, 0);
  
  for (int i = 0; i < len_features_; i++)
  {
    out.block<2,1>(dxZ+i*3,0) = q_feat_boxminus(Quat(x1.block<4,1>(xZ+i*5,0)), Quat(x2.block<4,1>(xZ+i*5,0)));
    out(dxZ+i*3+2) = x1(xZ+i*5+4) - x2(xZ+i*5+4);
  }
}

/**
 * @brief VIEKF::global_to_local_feature_id
 * @param global_id
 * @return the local id (the index of the feature in the state)
 */
int VIEKF::global_to_local_feature_id(const int global_id) const
{
  int i = 0;
  while ( i != features_.size() && features_[i].global_id != global_id)
  {
    i++;
  }
  if (i < features_.size())
  {
    return i;
  }
  else
  {
    return -1;
  }
}

/**
 * @brief VIEKF::fix_depth
 * Applies the inequality constraint per "Avoiding Negative Depth 
 * in Inverse Depth Bearing-Only SLAM" by Parsley and Julier
 */
void VIEKF::fix_depth()
{
  // Apply an Inequality Constraint per
  // "Avoiding Negative Depth in Inverse Depth Bearing-Only SLAM"
  // by Parsley and Julier
  for (int i = 0; i < len_features_; i++)
  {
    int xRHO_i = xZ + 5*i + 4;
    int dxRHO_i = dxZ + 3*i + 2;
    if (x_(xRHO_i, 0) != x_(xRHO_i, 0))
    {
      // if a depth state has gone NaN, reset it
      x_(xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
    if (x_(xRHO_i, 0) < 0.0)
    {
      // If the state has gone negative, reset it
      double err = 1.0/(2.0*min_depth_) - x_(xRHO_i, 0);
      P_(dxRHO_i, dxRHO_i) += err*err;
      x_(xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
    else if (x_(xRHO_i, 0) > 1e2)
    {
      // If the state has grown unreasonably large, reset it
      P_(dxRHO_i, dxRHO_i) = P0_feat_(2,2);
      x_(xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
  }
}

/**
 * @brief VIEKF::set_image
 * Builds the multilevel image from the supplied image, and saves it
 * as a class member variable
 * @param grey a greyscale image
 */
void VIEKF::set_image(const Mat &grey)
{
  // Build multilevel image
  ASSERT(grey.type() == CV_32FC1, "requires CV_32FC1 type image");
  
  img_[0] = grey;
  for (int i = 1; i < PYRAMID_LEVELS; i++)
  {
    pyrDown( img_[i-1], img_[i], Size( img_[i-1].cols/2, img_[i-1].rows/2 ) );
  }
}

/**
 * @brief VIEKF::inImage
 * Returns true if supplied pixel location is in image
 * @param pix
 * @return true if in image, false otherwise
 */
bool VIEKF::inImage(const Vector2f &pt) const
{
  return pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img_[0].cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img_[0].rows;
}

/**
 * @brief VIEKF::proj
 * Caculates the pixel corresponding to the bearing vector quaternion qz - requires that camera intrinsics have been set
 * @param qz Bearing Quaternion
 * @param eta (return value) pixel location
 * @param jac (return matrix) deta/dqz
 * @param calc_jac (default = true) calculate jacobian
 */
void VIEKF::proj(const Quat& qz, Vector2f& eta, Matrix2f& jac, bool calc_jac = true) const
{
  Vector3d zeta = qz.rota(e_z);
  double ezT_zeta = e_z.transpose() * zeta;
  
  eta = (cam_F_ * zeta / ezT_zeta + cam_center_).cast<float>();
  
  if (calc_jac)
  {
    Matrix3d sk_zeta = skew(zeta);
    MatrixXd T_z = T_zeta(qz);
    jac = (-cam_F_ * ((sk_zeta * T_z)/ezT_zeta - (zeta * e_z.transpose() * sk_zeta * T_z)/(ezT_zeta*ezT_zeta))).cast<float>();
  }
}


/**
 * @brief VIEKF::multiLvlPatch
 * Creates a multilevel patch vector from the multilevel image stored in the class
 * surrounding the pixel eta.
 * @param eta
 * @param dst
 */
void VIEKF::multiLvlPatch(const pixVector &eta, multiPatchVectorf& dst) const
{
  Size sz(PATCH_SIZE, PATCH_SIZE);
  float x = eta(0,0);
  float y = eta(1,0);
  for (int i = 0; i < PYRAMID_LEVELS; i++)
  {
    const Mat _dst(PATCH_SIZE, PATCH_SIZE, CV_32FC1, dst.data() + i * PATCH_SIZE*PATCH_SIZE, PATCH_SIZE * sizeof(float));
    getRectSubPix(img_[i], sz, Point2f(x,y), _dst);
    x /= 2.0;
    y /= 2.0;
  }
}

/**
 * @brief VIEKF::extractLvlfromPatch
 * Extracts a single level from the multilevel patch vector as a square patch matrix
 * @param src
 * @param level
 * @param dst
 */
void VIEKF::extractLvlfromPatch(multiPatchVectorf &src, const uint32_t level, patchMat &dst) const
{
  ASSERT(level < PYRAMID_LEVELS, "requested too large pyramid level");    
  dst = Eigen::Map<patchMat>((float*)src.data() + PATCH_SIZE*PATCH_SIZE*level);
}

/**
 * @brief VIEKF::multiLvlPatchSideBySide
 * builds an Eigen matrix of the patches levels side by side
 * @param src
 * @param dst
 */
void multiLvlPatchSideBySide(multiPatchVectorf& src, multiPatchMatrixf& dst)
{
  dst = Map<multiPatchMatrixf>(src.data());
}

/**
 * @brief drawPatch
 * Draws a transparent patch on an image (Requires a CV_8UC3 type image)
 * @param img - Image to draw on (CV_8UC3)
 * @param pt - Pixel location of center of patch
 * @param col - color
 * @param alpha - transparency (default = 0.3)
 */
void drawPatch(Mat& img, const Vector2f& pt, const Scalar& col, double alpha=0.3)
{
  ASSERT(img.type() == CV_8UC3, "Invalid image type supplied to drawPatch")
  if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img.cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img.rows)
  {
    cv::Mat roi = img(Rect(Point(pt(0,0), pt(1,0)), Size(PATCH_SIZE, PATCH_SIZE)));
    cv::Mat color(roi.size(), CV_8UC3, col); 
    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 
  }
}

/**
 * @brief VIEKF::multiLvlPatchToCv
 * Creates an opencv array of the patch levels so you can view it side-by-side
 * @param src
 * @param dst
 */
void multiLvlPatchToCv(multiPatchVectorf& src, Mat& dst)
{
  multiPatchMatrixf side_by_side;
  multiLvlPatchSideBySide(src, side_by_side);
  eigen2cv(side_by_side, dst);
}

}
