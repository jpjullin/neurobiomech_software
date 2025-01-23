#include "neurobio.h"

using namespace NEUROBIO_NAMESPACE;

int main() {
  auto &logger = utils::Logger::getInstance();
  logger.setLogLevel(utils::Logger::INFO);
  logger.info("Starting the application");

  try {
    auto devices = devices::Devices();
    devices.add(std::make_unique<devices::DelsysEmgDeviceMock>());
    devices.add(devices::MagstimRapidDeviceMock::findMagstimDevice());

    devices.connect();
    devices.startDataStreaming();
    devices.startRecording();
    logger.info("The system is now connected, streaming and recording data");

    std::this_thread::sleep_for(std::chrono::seconds(1));
    logger.info("The system has been recording data for 1 seconds");

    devices.stopRecording();
    devices.stopDataStreaming();
    devices.disconnect();
    logger.info(
        "The system has stopped recording data, streaming data and is now "
        "disconnected");

    const auto &dataCollectors = devices.getDataCollectors();
    for (auto &[deviceId, dataCollector] : dataCollectors) {
      logger.info("The device " + devices[deviceId].deviceName() +
                  " has collected " +
                  std::to_string(dataCollector->getTrialData().size()) +
                  " data points");
    }

  } catch (std::exception &e) {
    logger.fatal(e.what());
    return EXIT_FAILURE;
  }

  logger.info("Exiting the application");
  return EXIT_SUCCESS;
}