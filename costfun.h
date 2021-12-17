#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
Eigen::Affine3d orthogonize(const Eigen::Affine3d& p )
{
    Eigen::Matrix4d ret = p.matrix();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ret.block<3,3>(0,0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    double d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    Eigen::Matrix3d diag = Eigen::Matrix3d::Identity() * d;
    ret.block<3,3>(0,0) = svd.matrixU() * diag * svd.matrixV().transpose();
    return Eigen::Affine3d (ret);
}

Eigen::Matrix4d orthogonize(const Eigen::Matrix4d& p )
{
    Eigen::Matrix4d ret = p;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ret.block<3,3>(0,0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    double d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    Eigen::Matrix3d diag = Eigen::Matrix3d::Identity() * d;
    ret.block<3,3>(0,0) = svd.matrixU() * diag * svd.matrixV().transpose();
    return ret;
}


class LocalParameterizationSE3 : public ceres::LocalParameterization {
// adopted from https://github.com/strasdat/Sophus/blob/master/test/ceres/local_parameterization_se3.hpp
public:
    virtual ~LocalParameterizationSE3() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw,
                      double* T_plus_delta_raw) const {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);
        Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
        Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * Sophus::SE3d::exp(delta);
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // Dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const {
        Eigen::Map<Sophus::SE3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
                jacobian_raw);
        jacobian = T.Dx_this_mul_exp_x_at_0();
        return true;
    }

    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

class LocalParameterizationPlane : public ceres::LocalParameterization {
public:
    virtual ~LocalParameterizationPlane() {}

    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const {
        x_plus_delta[0] = x[0] + delta[0];
        x_plus_delta[1] = x[1] + delta[1];
        x_plus_delta[2] = x[2] + delta[2];
        x_plus_delta[3] = x[3] + delta[3];
        Eigen::Map<Eigen::Matrix<double, 3, 1>> x_plus_deltap (x_plus_delta);
        x_plus_deltap = x_plus_deltap / x_plus_deltap.norm();
        return true;
    }
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const {
        ceres::MatrixRef(jacobian_raw, 4, 4) = ceres::Matrix::Identity(4, 4);
        return true;
    }

    virtual int GlobalSize() const { return 4; }

    virtual int LocalSize() const { return 4; }
};

template<typename T> Sophus::SE3<T>  getSEFromParams(const T* const params)
{
//    Eigen::Map<const Eigen::Matrix<T,6,1>> eigen_laser_params(params);
//    Sophus::SE3<T> TT = Sophus::SE3<T>::exp(eigen_laser_params);
    Eigen::Map<Sophus::SE3<T> const> const TT(params);

    return TT;
}

struct costFunICPGT{
    const Sophus::SE3d imu_offset_inv;
    const Eigen::Vector4f local_point1;
    const Eigen::Vector4f gt_point2;


    costFunICPGT(const Eigen::Vector3f _local_point1, const Eigen::Vector3f & _gt_point2,Sophus::SE3d imu_offset) :
            local_point1(_local_point1.x(),_local_point1.y(),_local_point1.z(),1.0f),
            gt_point2(_gt_point2.x(),_gt_point2.y(),_gt_point2.z(),1.0f),
            imu_offset_inv(imu_offset.inverse())

    {}

    template <typename T>
    bool operator()(const T* const odom1tan,  const T* const instrument1tan,
                    T* residuals) const {


        Eigen::Map<Sophus::SE3<T> const>  pose1(odom1tan);
        Eigen::Map<Sophus::SE3<T> const>  instrument1pose(instrument1tan);
        Sophus::SE3<T> imu_pose1 = pose1 * imu_offset_inv.cast<T>();
        Eigen::Matrix<T,4,1> pt1 =imu_pose1 *instrument1pose *  local_point1.cast<T>();
        auto gt_point2T = gt_point2.cast<T>();
        residuals[0] = pt1.x()-gt_point2T.x();
        residuals[1] = pt1.y()-gt_point2T.y();
        residuals[2] = pt1.z()-gt_point2T.z();

        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector3f _local_point1, const Eigen::Vector3f & _gt_point2,Sophus::SE3d imu_offset) {
        return (new ceres::AutoDiffCostFunction<costFunICPGT,3, Sophus::SE3d::num_parameters,
                Sophus::SE3d::num_parameters>(
                new costFunICPGT(_local_point1, _gt_point2, imu_offset)));
    }
};


struct costFunICP{
    const Eigen::Vector4f local_point1;
    const Eigen::Vector4f local_point2;


    costFunICP(const Eigen::Vector4f _local_point1, const Eigen::Vector4f & _local_point2) :
            local_point1(_local_point1),local_point2(_local_point2)
    {}

    template <typename T>
    bool operator()(const T* const odom1tan, const T* const odom2tan,
                    T* residuals) const {


        Eigen::Map<Sophus::SE3<T> const>  pose1(odom1tan);
        Eigen::Map<Sophus::SE3<T> const>  pose2(odom2tan);



        Eigen::Matrix<T,4,1> pt1 =pose1 *  local_point1.cast<T>();
        Eigen::Matrix<T,4,1> pt2 =pose2 *  local_point2.cast<T>();

        residuals[0] = pt1.x()-pt2.x();
        residuals[1] = pt1.y()-pt2.y();
        residuals[2] = pt1.z()-pt2.z();


        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector4f _local_point1, const Eigen::Vector4f & _local_point2) {
        return (new ceres::AutoDiffCostFunction<costFunICP,3, Sophus::SE3d::num_parameters,Sophus::SE3d::num_parameters>(
                new costFunICP(_local_point1, _local_point2)));
    }
};

struct RelativePose{
    const Sophus::SE3d odom1;
    const Sophus::SE3d odom2;
    const Sophus::SE3d icrement_pose_measured;

    RelativePose(const  Sophus::SE3d& _odom1, const  Sophus::SE3d& _odom2 ) :
            odom1(_odom1),odom2(_odom2),
            icrement_pose_measured(odom1.inverse()*odom2)
    {}

    template <typename T>
    bool operator()(const T* const odom1tan, const T* const odom2tan,
                    T* residuals) const {

        Sophus::SE3<T> icrement_pose_measured_sophus(icrement_pose_measured.matrix().cast<T>());

        Eigen::Map<Sophus::SE3<T> const>  odom1(odom1tan);
        Eigen::Map<Sophus::SE3<T> const>  odom2(odom2tan);

        Sophus::SE3<T> increment = (odom1.inverse()*odom2);
        Eigen::Map<Eigen::Matrix<T,6,1>> residuals_map(residuals);
        residuals_map = (increment.log() - icrement_pose_measured_sophus.log());
        residuals_map[0] = residuals_map[0]*1e2;
        residuals_map[1] = residuals_map[1]*1e2;

        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3d& odom1,const Sophus::SE3d& odom2) {
        return (new ceres::AutoDiffCostFunction<RelativePose, 6,
                Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters >(
                new RelativePose(odom1, odom2)));
    }
};