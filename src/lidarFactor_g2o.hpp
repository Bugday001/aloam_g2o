// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h> //顶点类型

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>


class g2oLidarEdgeFactor : public g2o::BaseUnaryEdge<1, Eigen::Vector3d, g2o::VertexSE3>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	g2oLidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_) 
	: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	virtual void computeError()
	{
		const g2o::VertexSE3 *pose = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
		// measurement is p, point is p'
		// 四元数转为旋转矩阵--先归一化再转为旋转矩阵
		// Eigen::Matrix4d T = Eigen::Isometry3d(pose->estimate()).matrix();
		// Eigen::Matrix3d R = T.block<3, 3>(0, 0);
		// Eigen::Vector3d t_w_curr = T.block<3, 1>(0, 3);

		// Eigen::Vector3d lp = R * curr_point + t_w_curr;
		const Eigen::Isometry3d T = Eigen::Isometry3d(pose->estimate());
		Eigen::Vector3d lp = T * curr_point;
		Eigen::Matrix<double, 3, 1> nu = (lp - last_point_a).cross(lp - _measurement);
		Eigen::Matrix<double, 3, 1> de = last_point_a - _measurement;
		double nu_norm = nu.norm();
		_error(0, 0) = nu_norm/de.norm();
	}

	virtual void linearizeOplus() override
	{
		const g2o::VertexSE3 *v = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
		const Eigen::Isometry3d T = Eigen::Isometry3d(v->estimate());
		//  de/T的李代数
		Eigen::Vector3d w_axis = T * curr_point;
		Eigen::Matrix3d skew_lp;//  = Sophus::SO3d::hat(T * curr_point); //  左乘扰动
		skew_lp<< 0,          -w_axis(2), w_axis(1),
            w_axis(2),  0,          -w_axis(0),
            -w_axis(1), w_axis(0),  0;
		Eigen::Vector3d lp = T * curr_point;
		Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
		Eigen::Vector3d de = last_point_a - last_point_b;
		double de_norm = de.norm();
		Eigen::Matrix<double, 3, 6> dp_by_se3;
		(dp_by_se3.block<3, 3>(0, 0)).setIdentity();
		dp_by_se3.block<3, 3>(0, 3) = -skew_lp;
		w_axis = last_point_a - last_point_b;
		Eigen::Matrix3d skew_de;// = Sophus::SO3d::hat(last_point_a - last_point_b);
		skew_de<< 0,          -w_axis(2), w_axis(1),
            w_axis(2),  0,          -w_axis(0),
            -w_axis(1), w_axis(0),  0;
		_jacobianOplusXi.block<1, 6>(0, 0) = -nu.transpose() / nu.norm() * skew_de * dp_by_se3 / de_norm;
	}

	bool read(std::istream &in) {}
	bool write(std::ostream &out) const {}

protected:
	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

class g2oLidarPlaneFactor : public g2o::BaseUnaryEdge<1, Eigen::Vector3d, g2o::VertexSE3>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	g2oLidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_) 
	: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	virtual void computeError()
	{
		const g2o::VertexSE3 *pose = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
		// measurement is p, point is p'
		// 四元数转为旋转矩阵--先归一化再转为旋转矩阵
		Eigen::Matrix4d T = Eigen::Isometry3d(pose->estimate()).matrix();
		Eigen::Matrix3d R = T.block<3, 3>(0, 0);
		Eigen::Vector3d t_w_curr = T.block<3, 1>(0, 3);


		Eigen::Vector3d lp = R * curr_point + t_w_curr;
	
		_error(0, 0) = (lp - last_point_j).dot(ljm_norm);

	}

	bool read(std::istream &in) {}
	bool write(std::ostream &out) const {}

protected:
	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

class g2oLidarPlaneNormFactor : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	g2oLidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	virtual void computeError()
	{
		const g2o::VertexSE3 *pose = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
		// measurement is p, point is p'
		// 四元数转为旋转矩阵--先归一化再转为旋转矩阵
		// Eigen::Matrix4d T = Eigen::Isometry3d(pose->estimate()).matrix();
		// Eigen::Matrix3d R = T.block<3, 3>(0, 0);
		// Eigen::Vector3d t_w_curr = T.block<3, 1>(0, 3);

		// Eigen::Vector3d point_w = R * curr_point + t_w_curr;
		Eigen::Isometry3d T = Eigen::Isometry3d(pose->estimate());
		Eigen::Vector3d point_w = T * curr_point;
		_error(0, 0) = plane_unit_norm.dot(point_w) + _measurement;

	}

	virtual void linearizeOplus() override
	{
		const g2o::VertexSE3 *pose = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
		const Eigen::Isometry3d T = Eigen::Isometry3d(pose->estimate());
		//  de/T的李代数
		Eigen::Vector3d w_axis = T*curr_point;
		Eigen::Matrix3d skew_point_w;// = Sophus::SO3d::hat(T * c_p); //  左乘扰动
		skew_point_w<< 0,          -w_axis(2), w_axis(1),
            w_axis(2),  0,          -w_axis(0),
            -w_axis(1), w_axis(0),  0;
		Eigen::Matrix<double, 3, 6> dp_by_se3;
		(dp_by_se3.block<3, 3>(0, 0)).setIdentity();
		dp_by_se3.block<3, 3>(0, 3) = -skew_point_w;
		_jacobianOplusXi.block<1, 6>(0, 0) = plane_unit_norm.transpose() * dp_by_se3;
	}
	bool read(std::istream &in) {}
	bool write(std::ostream &out) const {}

protected:
	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};

