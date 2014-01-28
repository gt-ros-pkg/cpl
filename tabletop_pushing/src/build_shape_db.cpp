#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <ros/ros.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND

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

ShapeDescriptors getModelCentroids(ShapeDescriptors& sds, int num_clusters, std::vector<int>& cluster_ids)
{
  // cluster sds
  ShapeDescriptors centers;
  double min_err_change = 0.001;
  int max_iter = 1000;
  int num_retries = 10;
  bool normalize = false;
  clusterShapeFeatures(sds, num_clusters, cluster_ids,
                       centers, min_err_change, max_iter, num_retries, normalize);
  return centers;
}

ShapeDescriptors getModelCentroids(ShapeDescriptors& sds, int num_clusters)
{
  std::vector<int> cluster_ids;
  return getModelCentroids(sds, num_clusters, cluster_ids);
}

ShapeDescriptors getModelCentroids(std::vector<std::string> shape_paths, int num_clusters)
{
  ShapeDescriptors sds;
  for (int i = 0; i < shape_paths.size(); ++i)
  {
    ShapeDescriptors new_sds = readShapeDescriptors(shape_paths[i]);
    sds.insert(sds.end(), new_sds.begin(), new_sds.end());
  }
  return getModelCentroids(sds, num_clusters);
}

ShapeDescriptors getModelCentroids(std::string shape_path, int num_clusters)
{
  ShapeDescriptors sds = readShapeDescriptors(shape_path);
  return getModelCentroids(sds, num_clusters);
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

void writeExemplarsToShapeDBFile(std::string file_path, std::vector<std::string> exemplar_names,
                                 ShapeDescriptors& exemplars)
{
  std::ofstream shape_file(file_path.c_str()); //, std::ofstream::out | std::ofstream::app);
  ROS_INFO_STREAM("Writing examplers to file " << file_path);
  for (int i = 0; i < exemplars.size(); ++i)
  {
    ROS_INFO_STREAM("Writing exemplar for " << exemplar_names[i]);
    writeShapeDBLine(exemplar_names[i], exemplars[i], shape_file);
  }
  shape_file.close();
}


int mainBuildShapeDBFromSingleObjectFile(int argc, char** argv)
{
  if (argc < 6 || argc > 7)
  {
    ROS_ERROR_STREAM("Usage: " << argv[0] << " switch output_path dynamics_model_name shape_path num_clusters");
    return -1;
  }

  // Read in ouptut path name
  std::string output_path(argv[2]);

  // Read in dynamics model name
  std::string dynamics_model_name(argv[3]);

  // Read in file housing shape features
  std::string dynamics_model_shape_dir(argv[4]);

  int num_clusters = atoi(argv[5]);
  ROS_INFO_STREAM("num_clusters = " << num_clusters);

  // Read in shape descriptors associated with the building of the model
  ShapeDescriptors sds = getModelCentroids(dynamics_model_shape_dir, num_clusters);
  writeModelToShapeDBFile(output_path, dynamics_model_name, sds);
  return 0;
}

int mainBuildShpaeBasedObjectClusters(int argc, char** argv)
{
  int obj_class_start_idx = 6;
  if (argc < obj_class_start_idx)
  {
    ROS_ERROR_STREAM("need at least " << obj_class_start_idx << " number of params got " << argc);
    ROS_ERROR_STREAM("Usage: " << argv[0] <<
                     " switch output_path base_input_path num_clsuters shape_suffix obj_name0 [obj_name1, ...]");
    return -1;
  }
  std::string output_path(argv[2]);
  std::string base_path(argv[3]);
  int num_clusters = atoi(argv[4]);
  std::string shape_suffix(argv[5]);

  ROS_INFO_STREAM("num_clusters = " << num_clusters);

  std::vector<std::string> obj_class_names;
  std::vector<std::string> obj_class_paths;
  ShapeDescriptors class_exemplars;
  // Read in object class names
  for (int i = obj_class_start_idx; i < argc; ++i)
  {
    std::string obj_name(argv[i]);
    obj_class_names.push_back(obj_name);
    std::stringstream obj_class_path;
    obj_class_path << base_path << obj_name << shape_suffix;
    obj_class_paths.push_back(obj_class_path.str());

    // Build exemplar from the path
    ShapeDescriptors exemplar = getModelCentroids(obj_class_path.str(), 1);
    class_exemplars.push_back(exemplar[0]);
    // Write exemplars for each object class to disk
    // writeModelToShapeDBFile(output_path, obj_name, exemplar);
  }

  std::vector<int> cluster_ids;
  ShapeDescriptors centers = getModelCentroids(class_exemplars, num_clusters, cluster_ids);

  // Get names for combined classes
  std::vector<std::string> center_names;
  for (int i = 0; i < cluster_ids.size(); ++i)
  {
    center_names.push_back("objs");
  }
  for (int i = 0; i < cluster_ids.size(); ++i)
  {
    int cluster_idx = cluster_ids[i];
    std::string class_name = obj_class_names[i];
    ROS_INFO_STREAM("class " << class_name << " belongs to cluster " << cluster_ids[i]);
    std::stringstream center_name;
    center_name << center_names[cluster_idx] << "_" << class_name;
    center_names[cluster_idx] = center_name.str();
  }

  // Write resulting centers and labels to disk
  // Write center as well, with the long name, then parse accordingly in python and rebuild shape_dbs with
  // controller names
  writeExemplarsToShapeDBFile(output_path, center_names,  centers);
  return 0;
}

int main(int argc, char** argv)
{
  int seed = time(NULL);
  srand(seed);
  ROS_INFO_STREAM("Rand seed is: " << seed);
  // HACK:
  int run_switch = atoi(argv[1]);
  ROS_INFO_STREAM("Run_switch: " << run_switch);
  if (run_switch)
  {
    return mainBuildShapeDBFromSingleObjectFile(argc, argv);
  }
  else
  {

    return mainBuildShpaeBasedObjectClusters(argc, argv);
  }
}
