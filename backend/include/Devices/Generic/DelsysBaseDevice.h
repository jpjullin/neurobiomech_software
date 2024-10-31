#ifndef __STIMWALKER_DEVICES_DELSYS_BASE_DEVICE_H__
#define __STIMWALKER_DEVICES_DELSYS_BASE_DEVICE_H__

#include "stimwalkerConfig.h"

#include <array>
#include <asio.hpp>
#include <fstream>
#include <string>
#include <vector>

#include "Devices/Exceptions.h"
#include "Devices/Generic/AsyncDataCollector.h"
#include "Devices/Generic/AsyncDevice.h"
#include "Devices/Generic/TcpDevice.h"
#include "Utils/CppMacros.h"

namespace STIMWALKER_NAMESPACE::devices {
namespace DelsysBaseDeviceMock {
class CommandTcpDeviceMock;
class DataTcpDeviceMock;
} // namespace DelsysBaseDeviceMock

class DelsysCommands : public DeviceCommands {
public:
  DECLARE_DEVICE_COMMAND(START, 0);
  DECLARE_DEVICE_COMMAND(STOP, 1);

  virtual std::string toString() const {
    switch (m_Value) {
    case START:
      return START_AS_STRING + m_TerminaisonCharacters;
    case STOP:
      return STOP_AS_STRING + m_TerminaisonCharacters;
    default:
      throw UnknownCommandException("Unknown command in DelsysCommands");
    }
  }

  /// @brief The terminaison characters for the command ("\r\n\r\n")
  DECLARE_PROTECTED_MEMBER_NOGET(std::string, TerminaisonCharacters)

protected:
  friend class DelsysBaseDevice;
  DelsysCommands(int value)
      : m_TerminaisonCharacters("\r\n\r\n"), DeviceCommands(value) {}
  DelsysCommands() = delete;
};

class DelsysBaseDevice : public AsyncDevice, public AsyncDataCollector {
public:
  class CommandTcpDevice : public TcpDevice {
  public:
    CommandTcpDevice(const std::string &host, size_t port);
    CommandTcpDevice(const CommandTcpDevice &other) = delete;

    std::string deviceName() const override;

  protected:
    DeviceResponses parseAsyncSendCommand(const DeviceCommands &command,
                                          const std::any &data) override;
  };

  class DataTcpDevice : public TcpDevice {
  public:
    DataTcpDevice(const std::string &host, size_t port);
    DataTcpDevice(const DataTcpDevice &other) = delete;

    std::string deviceName() const override;

  protected:
    DeviceResponses parseAsyncSendCommand(const DeviceCommands &command,
                                          const std::any &data) override;
  };

public:
  /// @brief Constructor of the DelsysEmgDevice
  /// @param channelCount The number of channels of the device
  /// @param host The host (ip) of the device
  /// @param dataPort The port of the data device
  /// @param commandPort The port of the command device (default 50040)
  DelsysBaseDevice(size_t channelCount, const std::string &host,
                   size_t dataPort, size_t commandPort);
  DelsysBaseDevice(const DelsysBaseDevice &other) = delete;

protected:
  /// @brief Constructor of the DelsysEmgDevice that allows to pass mocker
  /// devices
  DelsysBaseDevice(std::unique_ptr<CommandTcpDevice> commandDevice,
                   std::unique_ptr<DataTcpDevice> dataDevice,
                   size_t channelCount);

public:
  /// @brief Destructor of the DelsysEmgDevice
  ~DelsysBaseDevice() = default;

protected:
  bool handleConnect() override;
  bool handleDisconnect() override;
  bool handleStartDataStreaming() override;
  bool handleStopDataStreaming() override;

  /// @brief The command device
  DECLARE_PROTECTED_MEMBER_NOGET(std::unique_ptr<CommandTcpDevice>,
                                 CommandDevice);

  /// @brief The data device
  DECLARE_PROTECTED_MEMBER_NOGET(std::unique_ptr<DataTcpDevice>, DataDevice);

  /// @brief Send a command to the [m_CommandDevice]
  /// @param command The command to send
  DeviceResponses parseAsyncSendCommand(const DeviceCommands &command,
                                        const std::any &data) override;

  /// DATA RELATED METHODS
public: // protected:
  void dataCheck() override;

protected:
  /// @brief The length of the data buffer for each channel (4 for the Delsys)
  DECLARE_PROTECTED_MEMBER(size_t, BytesPerChannel)

  /// @brief The snample count for each frame (27 for the Delsys)
  DECLARE_PROTECTED_MEMBER(size_t, SampleCount)

  /// @brief The buffer to read the data from the device
  DECLARE_PROTECTED_MEMBER(std::vector<char>, DataBuffer)
  void handleNewData(const data::DataPoint &data) override;
};

/// ------------ ///
/// MOCK SECTION ///
/// ------------ ///

namespace DelsysBaseDeviceMock {

class DelsysCommandsMock : public DelsysCommands {
public:
  DelsysCommandsMock(int value) : DelsysCommands(value) {}
  static DelsysCommandsMock fromString(const std::string &command) {
    if (command == DelsysCommandsMock(NONE).toString()) {
      return NONE;
    } else if (command == DelsysCommandsMock(START).toString()) {
      return START;
    } else if (command == DelsysCommandsMock(STOP).toString()) {
      return STOP;
    } else {
      throw UnknownCommandException("Unknown command in DelsysCommandsMock");
    }
  }

  DECLARE_DEVICE_COMMAND(NONE, -1);

  std::string toString() const override {
    switch (m_Value) {
    case NONE:
      return NONE_AS_STRING + m_TerminaisonCharacters;
    default:
      return DelsysCommands::toString();
    }
  }
};

class CommandTcpDeviceMock : public DelsysBaseDevice::CommandTcpDevice {
public:
  CommandTcpDeviceMock(const std::string &host, size_t port);
  bool write(const std::string &data) override;
  bool read(std::vector<char> &buffer) override;

protected:
  DECLARE_PROTECTED_MEMBER_NOGET(DelsysCommandsMock, LastCommand)

  bool handleConnect() override;
};

class DataTcpDeviceMock : public DelsysBaseDevice::DataTcpDevice {
public:
  DataTcpDeviceMock(size_t channelCount, const std::string &host, size_t port);
  bool read(std::vector<char> &buffer) override;

protected:
  DeviceResponses parseAsyncSendCommand(const DeviceCommands &command,
                                        const std::any &data) override;

  DECLARE_PROTECTED_MEMBER_NOGET(size_t, DataChannelCount);

  bool handleConnect() override;

  /// @brief The time at which the data started collecting
  DECLARE_PROTECTED_MEMBER_NOGET(
      std::chrono::time_point<std::chrono::system_clock>, StartTime)
};
}; // namespace DelsysBaseDeviceMock

} // namespace STIMWALKER_NAMESPACE::devices
#endif // __STIMWALKER_DEVICES_DELSYS_BASE_DEVICE_H__