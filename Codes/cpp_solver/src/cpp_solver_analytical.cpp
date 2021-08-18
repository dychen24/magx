#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <thread>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
const int kNumObservations = 8;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

// -------------
// pure C++ code
// -------------
struct Cost_FixedM_1mag {
  Cost_FixedM_1mag(double x, double y, double z, double x2, double y2,
                   double z2, double m)
      : Bx(x),
        By(y),
        Bz(z),
        Xs(x2),
        Ys(y2),
        Zs(z2),
        M{m} {}  // init the sensor position and the magnitude reading.
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const z,
                  const T *const theta, const T *const phy, const T *const Gx,
                  const T *const Gy, const T *const Gz, T *residual)
      const {  // x y z is the coordinates of magnate j, m is the attributes of
               // magate j, theta phy is the orientation of the magnate
    Eigen::Matrix<T, 3, 1> VecM =
        Eigen::Matrix<T, 3, 1>(sin(theta[0]) * cos(phy[0]),
                               sin(theta[0]) * sin(phy[0]), cos(theta[0])) *
        1e-7 * exp(M);
    Eigen::Matrix<T, 3, 1> VecR =
        Eigen::Matrix<T, 3, 1>(Xs - x[0], Ys - y[0], Zs - z[0]);
    T NormR = VecR.norm();
    Eigen::Matrix<T, 3, 1> B =
        (3.0 * VecR * (VecM.transpose() * VecR) / pow(NormR, 5) -
         VecM /
             pow(NormR, 3));  // convert it's unit to correspond with the input
    // std::cout << "B= " << (B(0, 0) + Gx[0]) * 1e6 << "\t" << (B(1, 0) +
    // Gy[0]) * 1e6 << "\t" << (B(2, 0) + Gz[0]) * 1e6 << "\n"; std::cout <<
    // B(0) << '\n'
    //           << B(1) << '\n'
    //           << B(2) << std::endl;
    residual[0] = (B(0, 0) + Gx[0]) * 1e6 - Bx;
    residual[1] = (B(1, 0) + Gy[0]) * 1e6 - By;
    residual[2] = (B(2, 0) + Gz[0]) * 1e6 - Bz;
    // std::cout << residual[0] << '\t' << residual[1] << '\t' << residual[2] <<
    // std::endl;
    return true;
  }

 private:
  const double Bx;
  const double By;
  const double Bz;
  const double Xs;
  const double Ys;
  const double Zs;
  const double M;
};

struct Cost_FixedM_2mag {
  Cost_FixedM_2mag(double Bx_, double By_, double Bz_, double Xs_, double Ys_,
                   double Zs_, double m)
      : Bx(Bx_),
        By(By_),
        Bz(Bz_),
        Xs(Xs_),
        Ys(Ys_),
        Zs(Zs_),
        M{m} {}  // init the sensor position and the magnitude reading.
  template <typename T>
  bool operator()(const T *const Gx, const T *const Gy, const T *const Gz,
                  const T *const x0, const T *const y0, const T *const z0,
                  const T *const theta0, const T *const phy0, const T *const x1,
                  const T *const y1, const T *const z1, const T *const theta1,
                  const T *const phy1, T *residual)
      const {  // x y z is the coordinates of magnate j, m is the attributes of
               // magate j, theta phy is the orientation of the magnate
    // mag one
    Eigen::Matrix<T, 3, 1> VecM0 =
        Eigen::Matrix<T, 3, 1>(sin(theta0[0]) * cos(phy0[0]),
                               sin(theta0[0]) * sin(phy0[0]), cos(theta0[0])) *
        1e-7 * exp(M);
    Eigen::Matrix<T, 3, 1> VecR0 =
        Eigen::Matrix<T, 3, 1>(Xs - x0[0], Ys - y0[0], Zs - z0[0]);
    T NormR0 = VecR0.norm();
    Eigen::Matrix<T, 3, 1> B0 =
        (3.0 * VecR0 * (VecM0.transpose() * VecR0) / pow(NormR0, 5) -
         VecM0 /
             pow(NormR0, 3));  // convert it's unit to correspond with the input
    // mag two
    Eigen::Matrix<T, 3, 1> VecM1 =
        Eigen::Matrix<T, 3, 1>(sin(theta1[0]) * cos(phy1[0]),
                               sin(theta1[0]) * sin(phy1[0]), cos(theta1[0])) *
        1e-7 * exp(M);
    Eigen::Matrix<T, 3, 1> VecR1 =
        Eigen::Matrix<T, 3, 1>(Xs - x1[0], Ys - y1[0], Zs - z1[0]);
    T NormR1 = VecR1.norm();
    Eigen::Matrix<T, 3, 1> B1 =
        (3.0 * VecR1 * (VecM1.transpose() * VecR1) / pow(NormR1, 5) -
         VecM1 /
             pow(NormR1, 3));  // convert it's unit to correspond with the input

    residual[0] = (B0(0, 0) + B1(0, 0) + Gx[0]) * 1e6 - Bx;
    residual[1] = (B0(1, 0) + B1(1, 0) + Gy[0]) * 1e6 - By;
    residual[2] = (B0(2, 0) + B1(2, 0) + Gz[0]) * 1e6 - Bz;
    // std::cout << residual[0] << '\t' << residual[1] << '\t' << residual[2] <<
    // std::endl;
    return true;
  }

 private:
  const double Bx;
  const double By;
  const double Bz;
  const double Xs;
  const double Ys;
  const double Zs;
  const double M;
};

std::vector<double> cal_Bi(double xs, double ys, double zs,
                           std::vector<double> param) {
  double x = param[4];
  double y = param[5];
  double z = param[6];
  double theta = param[7];
  double phy = param[8];
  double Gx = param[0];
  double Gy = param[1];
  double Gz = param[2];
  double M = param[3];

  Eigen::Matrix<double, 3, 1> VecM =
      Eigen::Matrix<double, 3, 1>(sin(theta) * cos(phy), sin(theta) * sin(phy),
                                  cos(theta)) *
      1e-7 * M;
  Eigen::Matrix<double, 3, 1> VecR =
      Eigen::Matrix<double, 3, 1>(xs - x, ys - y, zs - z);
  double NormR = VecR.norm();
  Eigen::Matrix<double, 3, 1> B =
      (3.0 * VecR * (VecM.transpose() * VecR) / pow(NormR, 5) -
       VecM / pow(NormR, 3));  // convert it's unit to correspond with the input

  std::vector<double> reading = {(B(0, 0) + Gx) * 1e6, (B(1, 0) + Gy) * 1e6,
                                 (B(2, 0) + Gz) * 1e6};
  return reading;
}

class MagCost : public ceres::SizedCostFunction<3, 1, 1, 1, 1, 1, 1, 1, 1> {
 public:
  MagCost(const double Bx, const double By, const double Bz, const double Xs,
          const double Ys, const double Zs, const double M)
      : Bx_(Bx), By_(By), Bz_(Bz), Xs_(Xs), Ys_(Ys), Zs_(Zs), M_(M) {}
  virtual ~MagCost() {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    double x = parameters[0][0];
    double y = parameters[1][0];
    double z = parameters[2][0];
    double theta = parameters[3][0];
    double phy = parameters[4][0];
    double Gx = parameters[5][0];
    double Gy = parameters[6][0];
    double Gz = parameters[7][0];

    Eigen::Matrix<double, 3, 1> VecM =
        Eigen::Matrix<double, 3, 1>(sin(theta) * cos(phy),
                                    sin(theta) * sin(phy), cos(theta)) *
        1e-7 * exp(M_);
    Eigen::Matrix<double, 3, 1> VecR =
        Eigen::Matrix<double, 3, 1>(Xs_ - x, Ys_ - y, Zs_ - z);
    double NormR = VecR.norm();
    Eigen::Matrix<double, 3, 1> B =
        (3.0 * VecR * (VecM.transpose() * VecR) / pow(NormR, 5) -
         VecM /
             pow(NormR, 3));  // convert it's unit to correspond with the input

    residuals[0] = (B(0, 0) + Gx) * 1e6 - Bx_;
    residuals[1] = (B(1, 0) + Gy) * 1e6 - By_;
    residuals[2] = (B(2, 0) + Gz) * 1e6 - Bz_;

    if (!jacobians) return true;

    // calculate dx
    double t1 = VecM.transpose() * VecR;
    jacobians[0][0] =
        1e6 * (5 * 3 * VecR(0, 0) * VecR(0, 0) * t1 / pow(NormR, 7) -
               (2 * 3 * VecR(0, 0) * VecM(0, 0) + 3 * t1) / pow(NormR, 5));
    jacobians[0][1] =
        1e6 * (5 * 3 * VecR(0, 0) * VecR(1, 0) * t1 / pow(NormR, 7) -
               (3 * VecR(0, 0) * VecM(1, 0) + 3 * VecR(1, 0) * VecM(0, 0)) /
                   pow(NormR, 5));
    jacobians[0][2] =
        1e6 * (5 * 3 * VecR(0, 0) * VecR(2, 0) * t1 / pow(NormR, 7) -
               (3 * VecR(0, 0) * VecM(2, 0) + 3 * VecR(2, 0) * VecM(0, 0)) /
                   pow(NormR, 5));

    // calculate dy
    jacobians[1][0] =
        1e6 * (5 * 3 * VecR(0, 0) * VecR(1, 0) * t1 / pow(NormR, 7) -
               (3 * VecR(0, 0) * VecM(1, 0) + 3 * VecR(1, 0) * VecM(0, 0)) /
                   pow(NormR, 5));
    jacobians[1][1] =
        1e6 * (5 * 3 * VecR(1, 0) * VecR(1, 0) * t1 / pow(NormR, 7) -
               (2 * 3 * VecR(1, 0) * VecM(1, 0) + 3 * t1) / pow(NormR, 5));
    jacobians[1][2] =
        1e6 * (5 * 3 * VecR(1, 0) * VecR(2, 0) * t1 / pow(NormR, 7) -
               (3 * VecR(1, 0) * VecM(2, 0) + 3 * VecR(2, 0) * VecM(1, 0)) /
                   pow(NormR, 5));

    // calculate dz
    jacobians[2][0] =
        1e6 * (5 * 3 * VecR(0, 0) * VecR(2, 0) * t1 / pow(NormR, 7) -
               (3 * VecR(0, 0) * VecM(2, 0) + 3 * VecR(2, 0) * VecM(0, 0)) /
                   pow(NormR, 5));
    jacobians[2][1] =
        1e6 * (5 * 3 * VecR(1, 0) * VecR(2, 0) * t1 / pow(NormR, 7) -
               (3 * VecR(1, 0) * VecM(2, 0) + 3 * VecR(2, 0) * VecM(1, 0)) /
                   pow(NormR, 5));
    jacobians[2][2] =
        1e6 * (5 * 3 * VecR(2, 0) * VecR(2, 0) * t1 / pow(NormR, 7) -
               (2 * 3 * VecR(2, 0) * VecM(2, 0) + 3 * t1) / pow(NormR, 5));

    // calculate d(theta)
    double t2 = 1e-7 * exp(M_) *
                (VecR(0, 0) * cos(phy) * cos(theta) +
                 VecR(1, 0) * sin(phy) * cos(theta) - VecR(2, 0) * sin(theta));
    jacobians[3][0] =
        1e6 * (3 * VecR(0, 0) * t2 / pow(NormR, 5) -
               1e-7 * exp(M_) * cos(phy) * cos(theta) / pow(NormR, 3));
    jacobians[3][1] =
        1e6 * (3 * VecR(1, 0) * t2 / pow(NormR, 5) -
               1e-7 * exp(M_) * sin(phy) * cos(theta) / pow(NormR, 3));
    jacobians[3][2] = 1e6 * (3 * VecR(2, 0) * t2 / pow(NormR, 5) +
                             1e-7 * exp(M_) * sin(theta) / pow(NormR, 3));

    // calculate d(phi)
    double t3 = 1e-7 * exp(M_) *
                (-VecR(0, 0) * sin(phy) * sin(theta) +
                 VecR(1, 0) * sin(theta) * cos(phy));
    jacobians[4][0] =
        1e6 * (3 * VecR(0, 0) * t3 / pow(NormR, 5) +
               1e-7 * exp(M_) * sin(phy) * sin(theta) / pow(NormR, 3));
    jacobians[4][1] =
        1e6 * (3 * VecR(1, 0) * t3 / pow(NormR, 5) -
               1e-7 * exp(M_) * cos(phy) * sin(theta) / pow(NormR, 3));
    jacobians[4][2] = 1e6 * (3 * VecR(2, 0) * t3 / pow(NormR, 5));

    // calculate dG
    jacobians[5][0] = 1e6;
    jacobians[5][1] = 0;
    jacobians[5][2] = 0;
    jacobians[6][0] = 0;
    jacobians[6][1] = 1e6;
    jacobians[6][2] = 0;
    jacobians[7][0] = 0;
    jacobians[7][1] = 0;
    jacobians[7][2] = 1e6;

    return true;
  }

 private:
  const double Bx_;
  const double By_;
  const double Bz_;
  const double Xs_;
  const double Ys_;
  const double Zs_;
  const double M_;
};

std::vector<double> solve_1mag(std::vector<double> readings,
                               std::vector<double> pSensor,
                               std::vector<double> init_param) {
  // std::vector<float> test_vector = { 2,1,3 };
  // Eigen::MatrixXf readings_vec = Eigen::Map<Eigen::Matrix<double, 8, 3>
  // >(readings.data()); Eigen::MatrixXf pSensor_vec =
  // Eigen::Map<Eigen::Matrix<double, 8, 3> >(pSensor.data());
  Eigen::VectorXd readings_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      readings.data(), readings.size());
  Eigen::VectorXd pSensor_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      pSensor.data(), pSensor.size());
  // Eigen::MatrixXd readings_vec_1(&readings[0], 8, 3);
  // Eigen::MatrixXd pSensor_vec_1(&pSensor[0], 8, 3);
  // Eigen::Map<Eigen::MatrixXd> readings_vec(readings_vec_1.data(), 3, 8);
  // Eigen::Map<Eigen::MatrixXd> pSensor_vec(pSensor_vec_1.data(), 3, 8);
  // readings_vec = readings_vec.transpose();
  // pSensor_vec = pSensor_vec.transpose();
  // std::cout
  //     << "readings_vec: " << readings_vec << "\n";
  // std::cout << "pSensor_vec: " << pSensor_vec << "\n";

  double Gx = init_param[0];
  double Gy = init_param[1];
  double Gz = init_param[2];
  double m = init_param[3];
  double x = init_param[4];
  double y = init_param[5];
  double z = init_param[6];
  double theta = init_param[7];
  double phy = init_param[8];
  // std::cout << "Initial x: " << x << " y: " << y << " z: " << z << " m: " <<
  // m << " theta: " << theta << " phy: " << phy << " Gx: " << Gx << " Gy: " <<
  // Gy << " Gz: " << Gz << "\n";
  Problem problem;
  for (int i = 0; i < int(pSensor_vec.size() / 3); ++i) {
    // problem.AddResidualBlock(
    //     new AutoDiffCostFunction<Cost, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
    //         new Cost(testdata(i, 0), testdata(i, 1), testdata(i, 2),
    //         sPosition(i, 0), sPosition(i, 1), sPosition(i, 2))),
    //     NULL, &x, &y, &z, &m, &theta, &phy, &Gx, &Gy, &Gz);

    problem.AddResidualBlock(
        new AutoDiffCostFunction<Cost_FixedM_1mag, 3, 1, 1, 1, 1, 1, 1, 1, 1>(
            new Cost_FixedM_1mag(readings_vec[i * 3], readings_vec[i * 3 + 1],
                                 readings_vec[i * 3 + 2], pSensor_vec[i * 3],
                                 pSensor_vec[i * 3 + 1], pSensor_vec[i * 3 + 2],
                                 m)),
        NULL, &x, &y, &z, &theta, &phy, &Gx, &Gy, &Gz);
  }
  Solver::Options options;
  // options.max_num_iterations = 1e6;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = std::thread::hardware_concurrency();
  // options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.max_num_iterations = 1e5;
  // options.min_relative_decrease = 1e-16;
  // options.max_num_consecutive_invalid_steps = 1e6;
  // options.function_tolerance = 1e-32;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << "\n";
  // std::cout << "Initial x: " << 0.0 << " y: " << 0.0 << " z: " << 0.0 << " m:
  // " << 0.0 << " theta: " << 0.0 << " phy: " << 0.0 << "\n"; std::cout <<
  // "Final x: " << x << " y: " << y << " z: " << z << " m: " << m << " theta: "
  // << theta << " phy: " << phy << " Gx: " << Gx << " Gy: " << Gy << " Gz: " <<
  // Gz << "\n";

  // set params
  std::vector<double> result_vec = {Gx, Gy, Gz, m, x, y, z, theta, phy};
  return result_vec;
}

std::vector<double> solve_2mag(std::vector<double> readings,
                               std::vector<double> pSensor,
                               std::vector<double> init_param) {
  // std::vector<float> test_vector = { 2,1,3 };
  // Eigen::MatrixXf readings_vec = Eigen::Map<Eigen::Matrix<double, 8, 3>
  // >(readings.data()); Eigen::MatrixXf pSensor_vec =
  // Eigen::Map<Eigen::Matrix<double, 8, 3> >(pSensor.data());
  Eigen::VectorXd readings_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      readings.data(), readings.size());
  Eigen::VectorXd pSensor_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      pSensor.data(), pSensor.size());
  // Eigen::MatrixXd readings_vec_1(&readings[0], 8, 3);
  // Eigen::MatrixXd pSensor_vec_1(&pSensor[0], 8, 3);
  // Eigen::Map<Eigen::MatrixXd> readings_vec(readings_vec_1.data(), 3, 8);
  // Eigen::Map<Eigen::MatrixXd> pSensor_vec(pSensor_vec_1.data(), 3, 8);
  // readings_vec = readings_vec.transpose();
  // pSensor_vec = pSensor_vec.transpose();
  // std::cout
  //     << "readings_vec: " << readings_vec << "\n";
  // std::cout << "pSensor_vec: " << pSensor_vec << "\n";

  double Gx = init_param[0];
  double Gy = init_param[1];
  double Gz = init_param[2];
  double m = init_param[3];
  double x0 = init_param[4];
  double y0 = init_param[5];
  double z0 = init_param[6];
  double theta0 = init_param[7];
  double phy0 = init_param[8];
  double x1 = init_param[9];
  double y1 = init_param[10];
  double z1 = init_param[11];
  double theta1 = init_param[12];
  double phy1 = init_param[13];

  // std::cout << "Initial x: " << x << " y: " << y << " z: " << z << " m: " <<
  // m << " theta: " << theta << " phy: " << phy << " Gx: " << Gx << " Gy: " <<
  // Gy << " Gz: " << Gz << "\n";
  Problem problem;
  for (int i = 0; i < int(pSensor_vec.size() / 3); ++i) {
    // problem.AddResidualBlock(
    //     new AutoDiffCostFunction<Cost, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
    //         new Cost(testdata(i, 0), testdata(i, 1), testdata(i, 2),
    //         sPosition(i, 0), sPosition(i, 1), sPosition(i, 2))),
    //     NULL, &x, &y, &z, &m, &theta, &phy, &Gx, &Gy, &Gz);

    problem.AddResidualBlock(
        new AutoDiffCostFunction<Cost_FixedM_2mag, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1>(new Cost_FixedM_2mag(
            readings_vec[i * 3], readings_vec[i * 3 + 1],
            readings_vec[i * 3 + 2], pSensor_vec[i * 3], pSensor_vec[i * 3 + 1],
            pSensor_vec[i * 3 + 2], m)),
        NULL, &Gx, &Gy, &Gz, &x0, &y0, &z0, &theta0, &phy0, &x1, &y1, &z1,
        &theta1, &phy1);
  }
  Solver::Options options;
  // options.max_num_iterations = 1e6;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = std::thread::hardware_concurrency();
  // options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.max_num_iterations = 1e5;
  // options.min_relative_decrease = 1e-16;
  // options.max_num_consecutive_invalid_steps = 1e6;
  // options.function_tolerance = 1e-32;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << "\n";
  // std::cout << "Initial x: " << 0.0 << " y: " << 0.0 << " z: " << 0.0 << " m:
  // " << 0.0 << " theta: " << 0.0 << " phy: " << 0.0 << "\n"; std::cout <<
  // "Final x: " << x << " y: " << y << " z: " << z << " m: " << m << " theta: "
  // << theta << " phy: " << phy << " Gx: " << Gx << " Gy: " << Gy << " Gz: " <<
  // Gz << "\n";

  // set params
  std::vector<double> result_vec = {Gx,     Gy,   Gz, m,  x0, y0,     z0,
                                    theta0, phy0, x1, y1, z1, theta1, phy1};
  return result_vec;
}

std::vector<double> calB(std::vector<double> pSensor,
                         std::vector<double> init_param) {
  Eigen::VectorXd pSensor_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      pSensor.data(), pSensor.size());

  double Gx = init_param[0];
  double Gy = init_param[1];
  double Gz = init_param[2];
  double m = init_param[3];
  double x = init_param[4];
  double y = init_param[5];
  double z = init_param[6];
  double theta = init_param[7];
  double phy = init_param[8];

  std::vector<double> result, Bi;

  for (int i = 0; i < int(pSensor_vec.size() / 3); ++i) {
    Bi = cal_Bi(pSensor_vec[i * 3], pSensor_vec[i * 3 + 1],
                pSensor_vec[i * 3 + 2], init_param);
    result.insert(result.end(), Bi.begin(), Bi.end());
  }

  // set params
  return result;
}

// wrap C++ function with NumPy array IO
py::array_t<double> py_solve_1mag(
    py::array_t<double, py::array::c_style | py::array::forcecast> readings,
    py::array_t<double, py::array::c_style | py::array::forcecast> pSensor,
    py::array_t<double, py::array::c_style | py::array::forcecast> init_param) {
  // allocate std::vector (to pass to the C++ function)
  std::vector<double> readings_vec(readings.size());
  std::vector<double> pSensor_vec(pSensor.size());
  std::vector<double> init_param_vec(init_param.size());

  std::vector<double> result_vec(init_param.size());
  // copy py::array -> std::vector
  std::memcpy(readings_vec.data(), readings.data(),
              readings.size() * sizeof(double));
  std::memcpy(pSensor_vec.data(), pSensor.data(),
              pSensor.size() * sizeof(double));
  std::memcpy(init_param_vec.data(), init_param.data(),
              init_param.size() * sizeof(double));

  // call pure C++ function
  result_vec = solve_1mag(readings_vec, pSensor_vec, init_param_vec);
  // std::vector<double> result_vec = multiply(array_vec);
  // multiply2(array_vec, result_vec);

  // allocate py::array (to pass the result of the C++ function to Python)
  auto result = py::array_t<double>(init_param.size());
  auto result_buffer = result.request();
  double *result_ptr = (double *)result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr, result_vec.data(),
              result_vec.size() * sizeof(double));

  return result;
}

py::array_t<double> py_solve_2mag(
    py::array_t<double, py::array::c_style | py::array::forcecast> readings,
    py::array_t<double, py::array::c_style | py::array::forcecast> pSensor,
    py::array_t<double, py::array::c_style | py::array::forcecast> init_param) {
  // allocate std::vector (to pass to the C++ function)
  std::vector<double> readings_vec(readings.size());
  std::vector<double> pSensor_vec(pSensor.size());
  std::vector<double> init_param_vec(init_param.size());

  std::vector<double> result_vec(init_param.size());
  // copy py::array -> std::vector
  std::memcpy(readings_vec.data(), readings.data(),
              readings.size() * sizeof(double));
  std::memcpy(pSensor_vec.data(), pSensor.data(),
              pSensor.size() * sizeof(double));
  std::memcpy(init_param_vec.data(), init_param.data(),
              init_param.size() * sizeof(double));

  // call pure C++ function
  result_vec = solve_2mag(readings_vec, pSensor_vec, init_param_vec);
  // std::vector<double> result_vec = multiply(array_vec);
  // multiply2(array_vec, result_vec);

  // allocate py::array (to pass the result of the C++ function to Python)
  auto result = py::array_t<double>(init_param.size());
  auto result_buffer = result.request();
  double *result_ptr = (double *)result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr, result_vec.data(),
              result_vec.size() * sizeof(double));

  return result;
}

py::array_t<double> py_calB(
    py::array_t<double, py::array::c_style | py::array::forcecast> pSensor,
    py::array_t<double, py::array::c_style | py::array::forcecast> init_param) {
  // allocate std::vector (to pass to the C++ function)
  std::vector<double> pSensor_vec(pSensor.size());
  std::vector<double> init_param_vec(init_param.size());

  std::vector<double> result_vec;
  // copy py::array -> std::vector
  std::memcpy(pSensor_vec.data(), pSensor.data(),
              pSensor.size() * sizeof(double));
  std::memcpy(init_param_vec.data(), init_param.data(),
              init_param.size() * sizeof(double));
  // call pure C++ function

  // std::vector<double> result_vec = multiply(array_vec);
  result_vec = calB(pSensor_vec, init_param_vec);

  // allocate py::array (to pass the result of the C++ function to Python)
  int result_size;
  if (init_param.size() == 9)
    result_size = pSensor_vec.size();
  else
    result_size = (pSensor_vec.size() / 3 + 1) * 3;
  // std::cout << result_size;
  auto result = py::array_t<double>(result_size);
  auto result_buffer = result.request();
  double *result_ptr = (double *)result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr, result_vec.data(),
              result_vec.size() * sizeof(double));

  return result;
}

PYBIND11_MODULE(cppsolver, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cppsolver

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  m.def(
      "solve_1mag", &py_solve_1mag,
      "solve using the given parameters, sensor readings and sensor positions");

  m.def(
      "solve_2mag", &py_solve_2mag,
      "solve using the given parameters, sensor readings and sensor positions");

  m.def("calB", &py_calB, "Cal B given psensor and params");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
