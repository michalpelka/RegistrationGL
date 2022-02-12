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


class LocalParameterizationSE32 : public ceres::LocalParameterization {
public:
    LocalParameterizationSE32() {}
    virtual ~LocalParameterizationSE32() {}
    bool Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

        Sophus::SE3d T = Sophus::SE3d::exp(lie);
        Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);
        Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T*T).log();

        for(int i = 0; i < 6; ++i) x_plus_delta[i] = x_plus_delta_lie(i, 0);

        return true;
    }
    bool ComputeJacobian(const double *x, double *jacobian) const {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

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

struct costFunICP2 :  public ceres::SizedCostFunction<3, 6, 6> {
    const Eigen::Vector4d local_point1;
    const Eigen::Vector4d local_point2;

    costFunICP2(const Eigen::Vector4f _local_point1, const Eigen::Vector4f & _local_point2) :
            local_point1(_local_point1.cast<double>()),local_point2(_local_point2.cast<double>())
    {}

    bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Sophus::Vector6d> params1(parameters[0]);
        Eigen::Map<const Sophus::Vector6d> params2(parameters[1]);

        const Sophus::SE3d pose1 = Sophus::SE3d::exp(params1);
        const Sophus::SE3d pose2 = Sophus::SE3d::exp(params2);

        const Eigen::Matrix<double,4,1> pt1 =pose1 *  local_point1;
        const Eigen::Matrix<double,4,1> pt2 =pose2 *  local_point2;

        residuals[0] = pt1.x()-pt2.x();
        residuals[1] = pt1.y()-pt2.y();
        residuals[2] = pt1.z()-pt2.z();

        if(jacobians != NULL)
        {
            if(jacobians[0] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J(jacobians[0]);
                J.setZero();
                Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jm;
                J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
                J.block<3,3>(0,3) = -Sophus::SO3d::hat(pt1.head<3>());
            }
            if(jacobians[1] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J(jacobians[1]);
                J.setZero();
                Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jm;
                J.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
                J.block<3,3>(0,3) = Sophus::SO3d::hat(pt2.head<3>());
            }
        }
        return true;
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