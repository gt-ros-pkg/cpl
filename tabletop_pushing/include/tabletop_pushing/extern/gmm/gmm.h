/**
 * Class for 3D Gaussian Mixture Model
 * @author Ahmad Humayun
 * @author nakazawa.atsushi
 **/

#ifndef GMM_H
#define GMM_H

#define _USE_MATH_DEFINES

#include <math.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

namespace fs = boost::filesystem;

#define NUM_GMM_DIMS 3
#define HALF_NUM_GMM_DIMS (float)NUM_GMM_DIMS/2
#define NUM_SIGMA_VALS (NUM_GMM_DIMS*(NUM_GMM_DIMS+1))/2

typedef cv::Vec<float, NUM_GMM_DIMS> GMMFloatPnt;
typedef cv::Vec<float, NUM_SIGMA_VALS> GMMSigmaVal;

class Gaussian {
 private:
  cv::Mat imat;
  double prob_norm;


  void adjustProbNorm() {
    prob_norm = pow(2 * M_PI, HALF_NUM_GMM_DIMS) * sqrt(determinant(mat));
  }


  void adjustForPSD() {
    // All covariance matrix should be positive semi-definite (PSD).
    // All eigenvalues of a PSD matrix are positive. After doing the
    // eigen-decomposition, all small or negative eigenvalues are converted
    // turned positive - and the covariance matrix constructed back

    const float MIN_EIGENVALUE = 1e-2;
    const float EIGENVALUE_SCALING = 0.1;

    // This function supposes that atleast one eigenvalue is larger than MIN_VAL

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(mat, eigenvalues, eigenvectors);

    // transpose so each column is a eigenvector
    eigenvectors = eigenvectors.t();

    bool eigenval_change = false;
    for (int d = 1; d < NUM_GMM_DIMS; d++) {
      eigenval_change = eigenval_change || eigenvalues.at<float>(d) < MIN_EIGENVALUE;
    }

    // if the last NUM_GMM_DIMS-1 smallest eigenvalues are negative
    if (eigenval_change) {
      // take a small fraction of the next smallest positive eigenvalue as the
      // current eigenvalue (just a heurestic)
      for (int d = 1; d < NUM_GMM_DIMS; d++) {
        if (eigenvalues.at<float>(d) < MIN_EIGENVALUE)
          eigenvalues.at<float>(d) = eigenvalues.at<float>(d-1) * EIGENVALUE_SCALING;
        if (eigenvalues.at<float>(d) < MIN_EIGENVALUE)
          eigenvalues.at<float>(d) = MIN_EIGENVALUE;
      }

      // make a matrix with eigenvalues on the diagonal
      cv::Mat eigval_mat = cv::Mat::zeros(NUM_GMM_DIMS, NUM_GMM_DIMS, CV_32F);
      for (int d = 0; d < NUM_GMM_DIMS; d++)
        eigval_mat.at<float>(d,d) = eigenvalues.at<float>(d);

      // reconstruct the covariance matrix
      mat = eigenvectors * eigval_mat * eigenvectors.inv();
    }
  }


 public:
  // The minimum variance value controls what is the resolving power of the
  // GMM for far-away points. Consider that the sigma is diagonal with values
  // t i.e. all dimensions have the variance t. Considering that exp(-500)
  // is the minimum non-zero value (we can even go further) a machine can
  // compute, then a sample point will only have non-zero probability if its
  // l_2 square distance is less than 0.5*500*t.
  static const float MIN_GMM_VARIANCE = 0.0;

  GMMFloatPnt mv;
  cv::Mat mat;

  Gaussian() {
    mat = cv::Mat::eye(NUM_GMM_DIMS, NUM_GMM_DIMS, CV_32F);
    imat = cv::Mat(mat.inv());

    // get the probability normalization value
    adjustProbNorm();

    GMMFloatPnt temp;
    for (int d = 0; d < NUM_GMM_DIMS; d++)
      temp[d] = std::numeric_limits<float>::quiet_NaN();
    SetMean(temp);
  }

  void SetMean(const GMMFloatPnt& mean_val) {
    mv = mean_val;
  }

  void SetSigma(const GMMSigmaVal& gmm_sigma_val) {
    // set all the values
    int curr_idx = 0;
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      for (int e = d; e < NUM_GMM_DIMS; e++) {
        mat.at<float>(d, e) = gmm_sigma_val[curr_idx];
        mat.at<float>(e, d) = gmm_sigma_val[curr_idx];
        ++curr_idx;
      }
    }

    // Note: variance values need to be positive

    // set the diagonal values
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      if (mat.at<float>(d, d) < MIN_GMM_VARIANCE)
        mat.at<float>(d, d) = MIN_GMM_VARIANCE;
    }

    // perturb matrix if close to negative definite
    adjustForPSD();

    imat = cv::Mat(mat.inv());

    // get the probability normalization value
    adjustProbNorm();
  }


  void dispparams() const {
    for (int d = 0; d < NUM_GMM_DIMS-1; d++)
      std::cout << boost::format("%8.3f|") % mv[d];
    std::cout << boost::format("%8.3f || ") % mv[NUM_GMM_DIMS-1];

    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      std::cout << boost::format("%8.3f|") % mat.at<float>(d, d);
    }

    for (int d = 0; d < NUM_GMM_DIMS-1; d++) {
      for (int e = d+1; e < NUM_GMM_DIMS; e++) {
        std::cout << boost::format("%8.3f|") % mat.at<float>(d, e);
      }
    }
  }


  static GMMSigmaVal returnFixedSigma(const float& sigma) {
    // get the fixed sigma value which diagonal values sigma and off-diagonal 0s
    GMMSigmaVal temp_sigma_pnt;
    int curr_idx = 0;

    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      for (int e = d; e < NUM_GMM_DIMS; e++) {
        if (d == e)
          temp_sigma_pnt[curr_idx] = sigma;
        else
          temp_sigma_pnt[curr_idx] = 0;
        ++curr_idx;
      }
    }

    return temp_sigma_pnt;
  }


  float density(const GMMFloatPnt& x) const;
  float mahal(const GMMFloatPnt& x) const;
  void serialize(std::ofstream& fd) const;
  void deserialize(std::ifstream& fd);
};


class GMM {
 public:
  int nk;           // number of mixtures
  float *w;         // mixture weights / coefficients
  Gaussian *kernel; // the actual Gaussian kernels (would be an array of size nk)
  double em_thresh; // the max difference between two EM iterations before stopping
  double max_iter;  // maximum number of iterations if GMM doesn't converge
  double min_iter;  // minimum number of iterations if GMM is going to run regardless


  GMM(const double em_threshold=5.0e-2, const int max_iterations=50,
      const int min_iterations=5) : nk(0), w(NULL), kernel(NULL), em_thresh(em_threshold),
    max_iter(max_iterations), min_iter(min_iterations)
  {
  }

  GMM(const GMM& x) : nk(x.nk), em_thresh(x.em_thresh), max_iter(x.max_iter), min_iter(x.min_iter)
  {
    alloc(nk);
    for (int i = 0; i < nk; ++i)
    {
      kernel[i] = x.kernel[i];
      w[i] = x.w[i];
    }
  }

  GMM& operator=(const GMM& x)
  {
    if (this != &x)
    {
      free();
      em_thresh = x.em_thresh;
      max_iter = x.max_iter;
      min_iter = x.min_iter;
      alloc(x.nk);
      for (int i = 0; i < nk; ++i)
      {
        kernel[i] = x.kernel[i];
        w[i] = x.w[i];
      }
    }
    return *this;
  }

  ~GMM() {
    // dtor
    free();
  }


  void alloc(const int number_of_kernel) {
    // deallocate any previously allocated memory
    free();

    nk = number_of_kernel;
    kernel = new Gaussian[nk];
    w = new float[nk];
    for (int i = 0; i < nk; i++) {
      w[i] = 1.0 / (float) nk;
    }
  }


  void free() {
    if (nk != 0) {
      if (w != NULL)
        delete[] w;
      if (kernel != NULL)
        delete[] kernel;

      nk = 0;
    }
  }


  float probability(const GMMFloatPnt& x) const {
    float val = 0.0;

    for (int i = 0; i < nk; i++) {
      val += (w[i] * kernel[i].density(x));
    }
    return val;
  }


  float grabCutLikelihood(const GMMFloatPnt& x) const {

    float min_dist = FLT_MAX;
    for (int i = 0; i < nk; i++)
    {
      float d_i = kernel[i].mahal(x);
      if (d_i < min_dist)
      {
        min_dist = d_i;
      }
    }
    return min_dist;
  }


  float minl2clustercenterdist(const GMMFloatPnt& x) const {
    float min_dist = std::numeric_limits<float>::max();
    float dist;

    for (int i = 0; i < nk; i++) {
      dist = 0;
      for (int d = 0; d < NUM_GMM_DIMS; d++) {
        dist += pow(kernel[i].mv[d] - x[d], 2);
      }

      if (dist < min_dist)
        min_dist = dist;
    }

    return pow(min_dist, 0.5);
  }


  void dispparams() const {
    std::cout << "kernel parameters\n";
    std::cout << "No.:   weight : ";
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      std::cout << boost::format("|   m%c   ") % static_cast<char>('a'+d);
    }
    std::cout << " ||                 gaussian parameters..\n";
    std::cout << "                ";
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      std::cout << "|        ";
    }
    std::cout << " || ";
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      std::cout << boost::format("  s_%d%d  |") % d % d;
    }

    for (int d = 1; d <= NUM_GMM_DIMS-1; d++) {
      for (int e = d+1; e <= NUM_GMM_DIMS; e++) {
        std::cout << boost::format("  s_%d%d  |") % d % e;
      }
    }
    std::cout << std::endl;

    for (int i = 0; i < nk; i++) {
      std::cout << boost::format("%3d: %8.5f : |") % i % w[i];
      kernel[i].dispparams();
      std::cout << std::endl;
    }
  }


  int which_kernel(const GMMFloatPnt& x) const;
  double learn(const std::vector<GMMFloatPnt>& pts);
  double GmmEm(const std::vector<GMMFloatPnt>& pts);
  void kmeansInit(const std::vector<GMMFloatPnt>& pts, const float sigma);
  void initkernels(const std::vector<GMMFloatPnt>& pts, const float sigma);
  void savegmm(const fs::path& savefilepath) const;
  void loadgmm(const fs::path& loadfilepath);
};

#endif
