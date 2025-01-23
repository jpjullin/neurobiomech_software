#ifndef __NEUROBIO_DEVICES_GENERIC_TCP_DEVICE_H__
#define __NEUROBIO_DEVICES_GENERIC_TCP_DEVICE_H__

#include "Devices/Generic/AsyncDevice.h"

namespace NEUROBIO_NAMESPACE::devices {

/// @brief A class representing a Tcp device
/// @note This class is only available on Windows and Linux
class TcpDevice : public AsyncDevice {
public:
  /// @brief Constructor
  /// @param host The host name of the device
  /// @param port The port number of the device
  /// @param keepAliveInterval The interval to keep the device alive
  TcpDevice(const std::string &host, size_t port,
            const std::chrono::microseconds &keepAliveInterval);
  TcpDevice(const TcpDevice &other) = delete;

  /// @brief Read data from the tcp device
  /// @param bufferSize The size of the buffer to read
  /// @return The data read from the device
  virtual std::vector<char> read(size_t bufferSize);

  /// @brief Read data from the tcp device
  /// @param buffer The buffer to read the data into
  /// @return True if the read was successful, false otherwise
  virtual bool read(std::vector<char> &buffer);

  /// @brief Write data to the tcp device
  /// @param data The data to write to the device
  /// @return True if the write was successful, false otherwise
  virtual bool write(const std::string &data);

protected:
  /// Protected members with Get accessors

  /// @brief Get the host name of the device
  /// @return The host name of the device
  DECLARE_PROTECTED_MEMBER(std::string, Host)

  /// @brief Get the port number of the device
  /// @return The port number of the device
  DECLARE_PROTECTED_MEMBER(size_t, Port)

  /// Protected members without Get accessors

  /// @brief Get the async context of the device
  /// @return The async context of the device
  DECLARE_PROTECTED_MEMBER_NOGET(asio::io_context, TcpContext)

  /// @brief Get the tcp socket of the device
  /// @return The tcp socket of the device
  DECLARE_PROTECTED_MEMBER_NOGET(asio::ip::tcp::socket, TcpSocket)

protected:
  bool handleConnect() override;
  bool handleDisconnect() override;
};

} // namespace NEUROBIO_NAMESPACE::devices
#endif // __NEUROBIO_DEVICES_GENERIC_TCP_DEVICE_H__