#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <ros/ros.h>

// libSVM
// #include <libsvm/svm.h>

using cpl_visual_features::ShapeDescriptors;
using cpl_visual_features::ShapeDescriptor;
using namespace::tabletop_pushing;

void writeShapeDBLine(std::string model_name, ShapeDescriptor& sd, std::ofstream& shape_file)
{
  shape_file << model_name << ":";
  for (int i = 0; i < sd.size(); ++i)
  {
    shape_file << sd[i];
    if (i < sd.size()-1)
    {
      shape_file << " ";
    }
    else
    {
      shape_file << "\n";
    }
  }
}

ShapeDescriptors readShapeDescriptors(std::string shape_path)
{
  ShapeDescriptors sds;
  std::ifstream shape_file(shape_path.c_str());
  while(shape_file.good())
  {
    std::string line;
    std::getline(shape_file, line);
    if (line.size() < 1)
    {
      continue;
    }
    std::stringstream line_stream;
    line_stream << line;
    // Get descriptor and file name from line
    ShapeDescriptor cur_sd;
    while (line_stream.good())
    {
      double x;
      line_stream >> x;
      if (line_stream.good())
      {
        cur_sd.push_back(x);
      }
    }
    sds.push_back(cur_sd);
    // ROS_INFO_STREAM("cur_sd.size() = " << cur_sd.size());
  }
  ROS_INFO_STREAM("Read in " << sds.size() << " shape descriptors");
  return sds;
}


ShapeDescriptors getModelCentroids(std::string shape_path, int num_clusters)
{
  // TODO: Read in raw shape descriptors
  ShapeDescriptors sds = readShapeDescriptors(shape_path);
  // cluster sds
  ShapeDescriptors centers;
  std::vector<int> cluster_ids;
  double min_err_change = 0.001;
  int max_iter = 1000;
  int num_retries = 10;
  bool normalize = false;
  clusterShapeFeatures(sds, num_clusters, cluster_ids,
                       centers, min_err_change, max_iter, num_retries, normalize);

  return centers;
}

void writeModelToShapeDBFile(std::string file_path, std::string dyn_name, ShapeDescriptors& sds)
{
  std::ofstream shape_file(file_path.c_str(), std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < sds.size(); ++i)
  {
    writeShapeDBLine(dyn_name, sds[i], shape_file);
  }
  shape_file.close();
}

int main(int argc, char** argv)
{
  if (argc < 5 || argc > 6)
  {
    ROS_ERROR_STREAM("Usage: " << argv[0] << " output_path dynamics_model_name shape_path num_clusters");
    return -1;
  }

  // Read in ouptut path name
  std::string output_path(argv[1]);

  // Read in dynamics model name
  std::string dynamics_model_name(argv[2]);

  // Read in file housing shape features
  std::string dynamics_model_shape_dir(argv[3]);

  int num_clusters = atoi(argv[4]);
  ROS_INFO_STREAM("num_clusters = " << num_clusters);
  
  // Read in shape descriptors associated with the building of the model
  ShapeDescriptors sds = getModelCentroids(dynamics_model_shape_dir, num_clusters);
  writeModelToShapeDBFile(output_path, dynamics_model_name, sds);
  return 0;
}
