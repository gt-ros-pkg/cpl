#include "gmm.h"


float Gaussian::density(const GMMFloatPnt& x) const {

  cv::Mat p(NUM_GMM_DIMS, 1, CV_32F);
  cv::Mat pt(1, NUM_GMM_DIMS, CV_32F);
  float val;

  for (unsigned int d = 0; d < NUM_GMM_DIMS; d++)
    p.at<float>(d, 0) = x[d] - mv[d];
  memcpy(pt.data, p.data, NUM_GMM_DIMS*sizeof(float));
  cv::Mat v(pt * imat * p);

  val = (float) exp(-0.5 * v.at<float>(0, 0)) / prob_norm;

  return val;
}

/**
 * Compute the mahalanobis distance between the given point x and the mean of the distribution
 *
 * @param x Sample point to evaluate
 *
 * @return Mahalanobis distance between x and mu, with Sigma
 */
float Gaussian::mahal(const GMMFloatPnt& x) const
{
  // Compute the probability of the point (x,y,z), given the mean and
  // covariance of the kernel - PRML eq (2.43)

  cv::Mat p(NUM_GMM_DIMS, 1, CV_32F);
  cv::Mat pt(1, NUM_GMM_DIMS, CV_32F);

  for (unsigned int d = 0; d < NUM_GMM_DIMS; d++)
    p.at<float>(d, 0) = x[d] - mv[d];
  memcpy(pt.data, p.data, NUM_GMM_DIMS*sizeof(float));
  cv::Mat v(pt * imat * p);
  return v.at<float>(0, 0);
}

void Gaussian::serialize(std::ofstream& fd) const {
  int num_dims = NUM_GMM_DIMS;

  fd.write(reinterpret_cast<const char *>(&num_dims), sizeof(int));

  for (int i = 0; i < NUM_GMM_DIMS; i++) {
    fd.write(reinterpret_cast<const char *>(&(mv[i])), sizeof(float));
  }

  for (int d = 0; d < NUM_GMM_DIMS; d++) {
    for (int e = d; e < NUM_GMM_DIMS; e++) {
      fd.write(reinterpret_cast<const char *>(&(mat.at<float>(d, e))), sizeof(float));
    }
  }
}


void Gaussian::deserialize(std::ifstream& fd) {
  int num_dims;

  fd.read(reinterpret_cast<char *>(&num_dims), sizeof(int));

  if (num_dims != NUM_GMM_DIMS)
    throw "The number of dimensions for the Gaussian loaded is inconsistent with NUM_GMM_DIMS";

  for (int i = 0; i < NUM_GMM_DIMS; i++) {
    fd.read(reinterpret_cast<char *>(&(mv[i])), sizeof(float));
  }

  for (int d = 0; d < NUM_GMM_DIMS; d++) {
    for (int e = d; e < NUM_GMM_DIMS; e++) {
      fd.read(reinterpret_cast<char *>(&(mat.at<float>(d, e))), sizeof(float));
      mat.at<float>(e, d) = mat.at<float>(d, e);
    }
  }

  imat = cv::Mat(mat.inv());

  // set the probability normalization value
  adjustProbNorm();
}


int GMM::which_kernel(const GMMFloatPnt& x) const {
  // find the kernel in the GMM where the point has the maximum posterior
  // probability

  int maxk = 0;
  float maxprob = kernel[0].density(x);
  float prob;

  for (int i = 1; i < nk; i++) {
    prob = kernel[i].density(x);
    if (maxprob < prob) {
      maxk = i;
      maxprob = prob;
    }
  }

  return maxk;
}


double GMM::learn(const std::vector<GMMFloatPnt>& pts) {
  // Run Expectation Maximization to learn the GMM
  // also returns the log likelihood normalized by the number of pts
  // The comments refer to PRML Chapt 9

  // temporary variables for the M step
  GMMFloatPnt km;
  GMMSigmaVal ks;
  float N_k;
  float x, y, z;
  double likelihood;
  int curr_idx;

  float weight[nk];                   // stores the new mixing coefficient
                                      // (eq 9.26) - first compute N_k (the
                                      // effective number of points assigned to
                                      // cluster k) and divide by N at end.
  float **prob = new float*[nk];      // stores responsibilities of each data
                                      // point for every Gaussian kernel (NxK)
  for (int i = 0; i < nk; i++)
    prob[i] = new float[pts.size()];
  float *px = new float[pts.size()];

  // compute the denominator for the responsibility function - eq (9.23)
  for (unsigned int i = 0; i < pts.size(); i++) {
    px[i] = probability(pts[i]);
  }

  // compute the E and M step for each kernel (parallelizable)
  for (int k = 0; k < nk; k++) {
    weight[k] = 0.0;

    // compute Expectation (E step)
    for (unsigned int i = 0; i < pts.size(); i++) {
      // compute the prior * likelihood (for the responsibility posterior)
      // i.e. the numerator in eq (9.23)
      prob[k][i] = w[k] * kernel[k].density(pts[i]);

      // divide only if evidence is non-zero - eq (9.23)
      if (px[i] > 0.0)
        prob[k][i] /= px[i];
      else
        prob[k][i] = 0.0;

      // keep a running tally of N_k - eq (9.27)
      weight[k] += prob[k][i];
    }

    // compute the mixing coefficient - eq (9.26)
    weight[k] /= (float) pts.size();

    // numerator of eq (9.26)
    N_k = pts.size() * weight[k];

    // reset values to 0.0
    for (unsigned int d = 0; d < NUM_GMM_DIMS; d++)
      km[d] = 0.0;
    for (unsigned int s = 0; s < NUM_SIGMA_VALS; s++)
      ks[s] = 0.0;

    // compute Maximization (M step)
    for (unsigned int i = 0; i < pts.size(); i++) {
      // accumulate the new mean - eq (9.24)
      for (unsigned int d = 0; d < NUM_GMM_DIMS; d++) {
        km[d] += prob[k][i] * pts[i][d];
      }

      // accumulate the squared parts of eq (9.25)
      curr_idx = 0;
      for (int d = 0; d < NUM_GMM_DIMS; d++) {
        for (int e = d; e < NUM_GMM_DIMS; e++) {
          ks[curr_idx] += prob[k][i] * pts[i][d] * pts[i][e];
          ++curr_idx;
        }
      }
    }

    // compute the new mean - eq (9.24)
    for (int d = 0; d < NUM_GMM_DIMS; d++)
      km[d] /= N_k;

    // compute the covariance matrix (Cov(X,Y) = E[XY] - E[X]E[Y]) - eq (9.25)
    // also normalize the covariance
    for (unsigned int s = 0; s < NUM_SIGMA_VALS; s++)
      ks[s] /= N_k;
    curr_idx = 0;
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      for (int e = d; e < NUM_GMM_DIMS; e++) {
        ks[curr_idx] -= km[d] * km[e];
        ++curr_idx;
      }
    }

    // Update mean and covariance matrix on this kernel
    kernel[k].SetMean(km);
    kernel[k].SetSigma(ks);
  }

  // normalize the mixing weight (so they sum to 1)
  // and store them into the the GMM model
  float sum = 0.0;

  for (int k = 0; k < nk; k++)
    sum += weight[k];
  for (int k = 0; k < nk; k++) {
    // move the mixing weight in the right direction
    //w[k] = 0.1 * w[k] + 0.9 * (weight[k] / sum);
    w[k] = weight[k] / sum;
  }

  // compute the log likelihood - eq (9.28) (parallelizable)
  double loglikelihood = 0.0;
  for (unsigned int i = 0; i < pts.size(); i++) {
    likelihood = probability(pts[i]);
    loglikelihood += log(likelihood);
  }
  loglikelihood /= pts.size();

  for (int i = 0; i < nk; i++)
    delete[] prob[i];

  delete[] prob;
  delete[] px;

  return loglikelihood;
}


double GMM::GmmEm(const std::vector<GMMFloatPnt>& pts) {
  double normloglikelihood = 0.0;
  double new_normloglikelihood = -std::numeric_limits<double>::max();
  double diff;
  int iteration = 0;

  std::cout << "EM diff threshold: " << em_thresh << std::endl;
  do {
    iteration++;

    // transfer old log likelihood value
    normloglikelihood = new_normloglikelihood;

    // EM step for GMM
    new_normloglikelihood = this->learn(pts);
    if (iteration == 1)
      diff = std::numeric_limits<double>::quiet_NaN();
    else
      diff = new_normloglikelihood - normloglikelihood;

    std::cout << "Log likelihood: iter " <<
        boost::format("%3d : %+8.4f | %+8.4f") % iteration %
        new_normloglikelihood % diff << "\n";

    // stop EM iterations only if difference between this and the last
    // iteration is less than a threshold, or the number of iterations
    // has increased a certain number (but run EM for atleast a certain
    // number of iterations)
  } while(iteration <= min_iter ||
      (diff > em_thresh && iteration < max_iter));

  if (iteration >= max_iter && new_normloglikelihood - normloglikelihood > em_thresh) {
    std::cout << "EM failed to converge after " << iteration << " iterations : "
              << new_normloglikelihood;
  }

  return new_normloglikelihood;
}


void GMM::kmeansInit(const std::vector<GMMFloatPnt>& pts, const float sigma) {
  // initialize the kernels by running kmeans

  GMMFloatPnt temp_pnt;

  const int NUM_RETRIES = 3;
  const int MAX_ITERS = 100;
  const float DIST_THRESH = 0.001;

  cv::Mat bestlabels;
  cv::Mat clustercenters;

  cv::Mat rgbmat(pts.size(), 1, CV_32FC(NUM_GMM_DIMS));

  // convert vector to a CV_32FC3 Mat
  for (int i = 0; i < rgbmat.rows; i++) {
    for (int d = 0; d < NUM_GMM_DIMS; d++)
      temp_pnt[d] = pts[i][d];

    rgbmat.at<GMMFloatPnt>(i) = temp_pnt;

    /*
    for (int j=0; j < NUM_GMM_DIMS; j++)
      std::cout << (float)pts[i][j] << ",";
    std::cout << std::endl;
    */
  }
  std::cout << "Number of rows: " << rgbmat.rows << std::endl;
  std::cout << "Number of cols: " << rgbmat.cols << std::endl;
  std::cout << "nk: " << nk << std::endl;

  // if the number of clusters needed is more than the number of points
  if (rgbmat.rows < nk || nk == 0) {
    // reduce the number of kernels (reallocate memory)
    alloc(rgbmat.rows);
    std::cout << "Reduced the number of kernels to " << nk << " because not enough data points available" << std::endl;
  }

  // TODO: expose epsilon to user
  cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, MAX_ITERS, 0.0001);

  // run kmeans
  kmeans(rgbmat, nk, bestlabels, termcrit, NUM_RETRIES, cv::KMEANS_RANDOM_CENTERS, clustercenters);

  // discard clusters whose centers are a repeat
  float dist;

  // // compare all cluster pairs to see if any two clusters are too close to each other
  // for (unsigned int i = 1; i < clustercenters.rows; i++) {
  //   //std::cout << "Cluster centers: " << clustercenters.rows << std::endl;

  //   for (unsigned int j = 0; j < i; j++) {
  //     // check if cluster centers are the same

  //     // compute L2 distance
  //     dist = 0.0;
  //     for (int d = 0; d < NUM_GMM_DIMS; d++)
  //       dist += pow(clustercenters.at<float>(i,d) - clustercenters.at<float>(j,d), 2);
  //     dist = sqrt(dist);
  //     //std::cout << i << "," << j << " dist: " << dist << std::endl;

  //     // if the two cluster centers are too close to each other
  //     if (dist <= DIST_THRESH) {
  //       // discard cluster i

  //       // (1) replace all pixels labelled i with label j
  //       for (unsigned int k=0; k < bestlabels.rows; k++) {
  //         if (bestlabels.at<unsigned int>(k) == i)
  //           bestlabels.at<unsigned int>(k) = j;
  //       }

  //       // NOTE: Could also be adjusting the mean here of cluster j, but since
  //       //  DIST_THRESH we would suppose it doesn't change by much

  //       // (2) shift cluster centers one down the array following i
  //       for (unsigned int k=i; k < clustercenters.rows-1; k++) {
  //         memcpy(clustercenters.data + clustercenters.step[0]*k,
  //                clustercenters.data + clustercenters.step[0]*(k+1),
  //                clustercenters.step[0]);
  //       }

  //       // (3) shrink size of cluster center array by 1 row
  //       clustercenters = clustercenters.rowRange(0, clustercenters.rows-1);
  //       i--;

  //       break;
  //     }
  //   }
  // }

  // if (clustercenters.rows < nk) {
  //   std::cout << "The number of clusters were reduced from " << nk << " to "
  //             << clustercenters.rows << std::endl;

  //   // reduce the number of kernels (reallocate memory)
  //   alloc(clustercenters.rows);
  // }

  // transfer the cluster means to the kernels. Plus the Gaussians have a fixed
  // variance in each direction and no covariance
  for (int i = 0; i < clustercenters.rows; i++) {
    for (int d = 0; d < NUM_GMM_DIMS; d++)
      temp_pnt[d] = clustercenters.at<float>(i,d);

    kernel[i].SetMean(temp_pnt);

    kernel[i].SetSigma(Gaussian::returnFixedSigma(sigma));
  }
}


void GMM::initkernels(const std::vector<GMMFloatPnt>& pts, const float sigma) {
  // initilize kernels by using ramdom sampling within outer boundary
  GMMFloatPnt xmin, xmax, temp_mean;

  for (int d = 0; d < NUM_GMM_DIMS; d++)
    xmin[d] = xmax[d] = pts[0][d];

  // find the max and min R/G/B values
  for (unsigned int i = 0; i < pts.size(); i++) {
    for (int d = 0; d < NUM_GMM_DIMS; d++) {
      if (pts[i][d] < xmin[d])
        xmin[d] = pts[i][d];
      if (pts[i][d] > xmax[d])
        xmax[d] = pts[i][d];
    }
  }

  time_t ti;

  time(&ti);
  srand(ti);

  // randomly initialize all kernels with means randomly distributed around
  // the range of R/G/B values. Plus the Gaussians have a fixed variance in
  // each direction and no covariance
  for (int i = 0; i < nk; i++) {
    for (int d = 0; d < NUM_GMM_DIMS; d++)
      temp_mean[d] = xmin[d] + (float)((xmax[d] - xmin[d]) * rand()) / (float) RAND_MAX;

    kernel[i].SetMean(temp_mean);
    kernel[i].SetSigma(Gaussian::returnFixedSigma(sigma));
  }
}


void GMM::savegmm(const fs::path& savefilepath) const {
  std::ofstream fd_gmm;

  fd_gmm.open(savefilepath.c_str(), std::ofstream::binary | std::ofstream::trunc);

  fd_gmm.write(reinterpret_cast<const char *>(&nk), sizeof(int));
  fd_gmm.write(reinterpret_cast<const char *>(&em_thresh), sizeof(double));
  fd_gmm.write(reinterpret_cast<const char *>(&max_iter), sizeof(double));
  fd_gmm.write(reinterpret_cast<const char *>(&min_iter), sizeof(double));
  fd_gmm.write(reinterpret_cast<const char *>(w), sizeof(float)*nk);

  // serialize all kernels
  for (int i = 0; i < nk; i++) {
    kernel[i].serialize(fd_gmm);
  }

  fd_gmm.close();
}


void GMM::loadgmm(const fs::path& loadfilepath) {
  std::ifstream fd_gmm;

  fd_gmm.open(loadfilepath.c_str(), std::ofstream::binary);

  // clear all data before proceeding
  free();

  // de-serialize all variables
  int num_kernels;
  fd_gmm.read(reinterpret_cast<char *>(&num_kernels), sizeof(int));
  fd_gmm.read(reinterpret_cast<char *>(&em_thresh), sizeof(double));
  fd_gmm.read(reinterpret_cast<char *>(&max_iter), sizeof(double));
  fd_gmm.read(reinterpret_cast<char *>(&min_iter), sizeof(double));

  alloc(num_kernels);

  fd_gmm.read(reinterpret_cast<char *>(w), sizeof(float)*nk);

  // de-serialize all kernels
  for (int i = 0; i < nk; i++) {
    kernel[i].deserialize(fd_gmm);
  }

  fd_gmm.close();
}
