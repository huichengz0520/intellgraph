
#include <algorithm>

#include "glog/logging.h"
#include "mnist_classifier.h"
#include "simple_classifier.h"
#include "node/node_headers.h"
#include "utility/random.h"
#include "utility/system.h"

using namespace std;
using namespace intellgraph;
using namespace Eigen;

int main(int argc, char* argv[]) {
  // Initialize Google's logging library.
  FLAGS_alsologtostderr = false;
  FLAGS_minloglevel = 2;
  std::string log_path(GetCWD());
  if (log_path.empty()) {
    // Stores log files in tmp/ if GetCWD() fails
    log_path = "tmp/";
  } else {
    log_path += "/logs";
  }

  fLS::FLAGS_log_dir = log_path;
  google::InitGoogleLogging(argv[0]);

  Example2::run();
  return 0;
}
