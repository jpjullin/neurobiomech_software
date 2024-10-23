#include "Devices/Concrete/DelsysEmgDevice.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "Data/DataPoint.h"
#include "Data/FixedTimeSeries.h"
#include "Devices/Exceptions.h"
#include "Utils/Logger.h"

using namespace STIMWALKER_NAMESPACE::devices;

std::function<std::unique_ptr<STIMWALKER_NAMESPACE::data::TimeSeries>()>
    timeSeriesGenerator = []() {
      return std::make_unique<STIMWALKER_NAMESPACE::data::FixedTimeSeries>(
          std::chrono::microseconds(500));
    };

DelsysEmgDevice::CommandTcpDevice::CommandTcpDevice(const std::string &host,
                                                    size_t port)
    : TcpDevice(host, port, std::chrono::milliseconds(1000)) {}

std::string DelsysEmgDevice::CommandTcpDevice::deviceName() const {
  return "DelsysCommandTcpDevice";
}

DeviceResponses DelsysEmgDevice::CommandTcpDevice::parseAsyncSendCommand(
    const DeviceCommands &command, const std::any &data) {
  auto commandAsDelsys = DelsysCommands(command.getValue());
  write(commandAsDelsys.toString());
  std::vector<char> response = read(128);
  return std::strncmp(response.data(), "OK", 2) == 0 ? DeviceResponses::OK
                                                     : DeviceResponses::NOK;
}

DelsysEmgDevice::DataTcpDevice::DataTcpDevice(const std::string &host,
                                              size_t port)
    : TcpDevice(host, port, std::chrono::milliseconds(1000)) {}

std::string DelsysEmgDevice::DataTcpDevice::deviceName() const {
  return "DelsysDataTcpDevice";
}

DeviceResponses DelsysEmgDevice::DataTcpDevice::parseAsyncSendCommand(
    const DeviceCommands &command, const std::any &data) {
  throw InvalidMethodException(
      "This method should not be called for DataTcpDevice");
}

DelsysEmgDevice::DelsysEmgDevice(const std::string &host, size_t commandPort,
                                 size_t dataPort)
    : m_CommandDevice(std::make_unique<CommandTcpDevice>(host, commandPort)),
      m_DataDevice(std::make_unique<DataTcpDevice>(host, dataPort)),
      m_BytesPerChannel(4), m_SampleCount(27),
      m_DataBuffer(std::vector<char>(16 * m_SampleCount * m_BytesPerChannel)),
      AsyncDevice(std::chrono::milliseconds(1000)),
      AsyncDataCollector(16, std::chrono::microseconds(1),
                         timeSeriesGenerator) {
  m_IgnoreTooSlowWarning = true;
}

DelsysEmgDevice::DelsysEmgDevice(
    std::unique_ptr<DelsysEmgDevice::CommandTcpDevice> commandDevice,
    std::unique_ptr<DelsysEmgDevice::DataTcpDevice> dataDevice,
    const std::string &host, size_t commandPort, size_t dataPort)
    : m_CommandDevice(std::move(commandDevice)),
      m_DataDevice(std::move(dataDevice)), m_BytesPerChannel(4),
      m_SampleCount(27),
      m_DataBuffer(std::vector<char>(16 * m_SampleCount * m_BytesPerChannel)),
      AsyncDevice(std::chrono::milliseconds(1000)),
      AsyncDataCollector(16, std::chrono::microseconds(1),
                         timeSeriesGenerator) {
  m_IgnoreTooSlowWarning = true;
}

DelsysEmgDevice::~DelsysEmgDevice() {
  stopDataCollectorWorkers();
  stopDeviceWorkers();
}

std::string DelsysEmgDevice::deviceName() const { return "DelsysEmgDevice"; }

std::string DelsysEmgDevice::dataCollectorName() const {
  return "DelsysEmgDataCollector";
}

bool DelsysEmgDevice::handleConnect() {
  m_CommandDevice->connect();
  if (!m_CommandDevice->getIsConnected()) {
    utils::Logger::getInstance().fatal(
        "The command device is not connected, did you start Trigno?");
    return false;
  }
  m_CommandDevice->read(128); // Consume the welcome message

  m_DataDevice->connect();
  if (!m_DataDevice->getIsConnected()) {
    utils::Logger::getInstance().fatal(
        "The data device is not connected, did you start Trigno?");
    m_CommandDevice->disconnect();
    return false;
  }

  return true;
}

bool DelsysEmgDevice::handleDisconnect() {
  if (m_IsStreamingData) {
    stopDataStreaming();
  }

  m_CommandDevice->disconnect();
  m_DataDevice->disconnect();

  return true;
}

bool DelsysEmgDevice::handleStartDataStreaming() {
  if (m_CommandDevice->send(DelsysCommands::START) != DeviceResponses::OK) {
    return false;
  }

  // Wait until the data starts streaming
  return m_DataDevice->read(m_DataBuffer);
}

bool DelsysEmgDevice::handleStopDataStreaming() {
  if (m_CommandDevice->send(DelsysCommands::STOP) != DeviceResponses::OK) {
    return false;
  }
  return true;
}

DeviceResponses
DelsysEmgDevice::parseAsyncSendCommand(const DeviceCommands &command,
                                       const std::any &data) {
  throw InvalidMethodException(
      "This method should not be called for DelsysEmgDevice");
}

void DelsysEmgDevice::dataCheck() {
  m_DataDevice->read(m_DataBuffer);

  std::vector<float> dataAsFloat(m_DataChannelCount * m_SampleCount);
  std::memcpy(dataAsFloat.data(), m_DataBuffer.data(),
              m_DataChannelCount * m_SampleCount * m_BytesPerChannel);

  // // Convert the data to double
  std::vector<std::vector<double>> dataPoints;
  for (int i = 0; i < m_SampleCount; i++) {
    std::vector<double> dataAsDouble(m_DataChannelCount);
    std::transform(dataAsFloat.begin() + i * m_DataChannelCount,
                   dataAsFloat.begin() + (i + 1) * m_DataChannelCount,
                   dataAsDouble.begin(),
                   [](float x) { return static_cast<double>(x); });

    dataPoints.push_back(dataAsDouble);
  }

  addDataPoints(dataPoints);
}

void DelsysEmgDevice::handleNewData(const data::DataPoint &data) {
  // Do nothing
}

/// ------------ ///
/// MOCK SECTION ///
/// ------------ ///

DelsysEmgDeviceMock::CommandTcpDeviceMock::CommandTcpDeviceMock(
    const std::string &host, size_t port)
    : m_LastCommand(DelsysCommandsMock::NONE), CommandTcpDevice(host, port) {}

bool DelsysEmgDeviceMock::CommandTcpDeviceMock::write(const std::string &data) {
  // Store the last command
  m_LastCommand = DelsysCommandsMock::fromString(data);
  return true;
}

bool DelsysEmgDeviceMock::CommandTcpDeviceMock::read(
    std::vector<char> &buffer) {
  // Prepare a response with bunch of \0 characters of length buffer.size()
  std::fill(buffer.begin(), buffer.end(), 0);

  switch (m_LastCommand.getValue()) {
  case DelsysCommandsMock::NONE: {
    // Write the welcome message
    std::string welcomeMessage =
        "Delsys Trigno System Digital Protocol Version 3.6.0 \r\n\r\n";
    std::copy(welcomeMessage.begin(), welcomeMessage.end(), buffer.begin());
    break;
  }
  case DelsysCommands::START:
  case DelsysCommands::STOP: {
    // Write the OK message
    std::string response = "OK\r\n\r\n";
    std::copy(response.begin(), response.end(), buffer.begin());
    break;
  }

  default: {
    throw InvalidMethodException("This command is not MOCK yet");
  }
  }

  return true;
}

bool DelsysEmgDeviceMock::CommandTcpDeviceMock::handleConnect() { return true; }

DelsysEmgDeviceMock::DataTcpDeviceMock::DataTcpDeviceMock(
    const std::string &host, size_t port)
    : DataTcpDevice(host, port) {}

bool DelsysEmgDeviceMock::DataTcpDeviceMock::read(std::vector<char> &buffer) {
  // Write the value float(1) to the buffer assuming 16 channels and 27 samples
  // with 4 bytes per channel

  size_t bytesPerChannel(4);
  size_t channelCount(16);
  size_t sampleCount(27);

  // Wait for the next cycle of data
  static size_t counter = 0;
  std::this_thread::sleep_until(
      m_StartTime + std::chrono::microseconds(500 * sampleCount * counter));

  // Copy the 4-byte representation of the float into the byte array
  unsigned char dataAsChar[4];
  for (size_t i = 0; i < sampleCount; i++) {
    float value = static_cast<float>(std::sin(
        static_cast<float>(counter * sampleCount + i) / 2000.0f * 2 * M_PI));
    std::memcpy(dataAsChar, &value, sizeof(float));
    for (size_t j = 0; j < channelCount; j++)
      std::copy(dataAsChar, dataAsChar + 4,
                buffer.begin() + i * bytesPerChannel * channelCount +
                    j * bytesPerChannel);
  }

  counter++;
  return true;
}

bool DelsysEmgDeviceMock::DataTcpDeviceMock::handleConnect() {
  m_StartTime = std::chrono::system_clock::now();
  return true;
}

DeviceResponses DelsysEmgDeviceMock::DataTcpDeviceMock::parseAsyncSendCommand(
    const DeviceCommands &command, const std::any &data) {
  return DeviceResponses::OK;
}

DelsysEmgDeviceMock::DelsysEmgDeviceMock(const std::string &host,
                                         size_t commandPort, size_t dataPort)
    : DelsysEmgDevice(
          std::make_unique<DelsysEmgDeviceMock::CommandTcpDeviceMock>(
              host, commandPort),
          std::make_unique<DelsysEmgDeviceMock::DataTcpDeviceMock>(host,
                                                                   dataPort),
          host, commandPort, dataPort) {}

bool DelsysEmgDeviceMock::handleConnect() {
  if (shouldFailToConnect) {
    // Simulate a failure to connect after few time
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return false;
  }
  return DelsysEmgDevice::handleConnect();
}

bool DelsysEmgDeviceMock::handleStartDataStreaming() {
  if (shouldFailToStartDataStreaming) {
    // Simulate a failure to connect after few time
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return false;
  }
  return DelsysEmgDevice::handleStartDataStreaming();
}
