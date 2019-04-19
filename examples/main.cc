#include "edge/edge_headers.h"
#include "graph/graphfxn.h"
#include "mnist_classifier.h"
#include "node/node_headers.h"
#include "simple_classifier.h"
#include "utility/system.h"


int main(int argc, char* argv[]) {
   // Initialize Google's logging library.
   FLAGS_alsologtostderr = true;
   FLAGS_minloglevel = 1;
   std::string log_path(GetCWD());
   if (log_path.empty()) {
      // Stores log files in tmp/ if GetCWD() fails
      log_path = "tmp/";
   } else {
      log_path += "/logs";
   }

   fLS::FLAGS_log_dir = log_path;
   google::InitGoogleLogging(argv[0]);
   Example1::run();
   return 0;
}