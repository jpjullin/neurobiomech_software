#include "Server/TcpServer.h"

#include "Utils/Logger.h"
#include <asio/steady_timer.hpp>
#include <thread>

#include "Devices/Concrete/DelsysAnalogDevice.h"
#include "Devices/Concrete/DelsysEmgDevice.h"
#include "Devices/Concrete/MagstimRapidDevice.h"
#include "Devices/Generic/DelsysBaseDevice.h"
#include "Devices/Generic/Device.h"

using namespace NEUROBIO_NAMESPACE::server;

const size_t BYTES_IN_CLIENT_PACKET_HEADER = 8;
const size_t BYTES_IN_SERVER_PACKET_HEADER = 16;

// Here are the names of the devices that can be connected (for internal use)
const std::string DEVICE_NAME_DELSYS_EMG = "DelsysEmgDevice";
const std::string DEVICE_NAME_DELSYS_ANALOG = "DelsysAnalogDevice";
const std::string DEVICE_NAME_MAGSTIM = "MagstimRapidDevice";

TcpServer::TcpServer(int commandPort, int responsePort, int liveDataPort)
    : m_IsClientConnecting(false), m_IsServerRunning(false),
      m_CommandPort(commandPort), m_ResponsePort(responsePort),
      m_LiveDataPort(liveDataPort),
      m_TimeoutPeriod(std::chrono::milliseconds(5000)), m_ProtocolVersion(1) {};

TcpServer::~TcpServer() {
  if (m_IsServerRunning) {
    stopServer();
  }

  m_CommandAcceptor.reset();
  m_ResponseAcceptor.reset();
  m_LiveDataAcceptor.reset();
}

void TcpServer::startServer() {
  m_ServerWorker = std::thread([this]() { startServerSync(); });
}

void TcpServer::startServerSync() {
  auto &logger = utils::Logger::getInstance();

  // Create the contexts and acceptors
  m_CommandAcceptor = std::make_unique<asio::ip::tcp::acceptor>(
      m_Context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), m_CommandPort));
  logger.info("TCP Command server started on port " +
              std::to_string(m_CommandPort));

  m_ResponseAcceptor = std::make_unique<asio::ip::tcp::acceptor>(
      m_Context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), m_ResponsePort));
  logger.info("TCP Response server started on port " +
              std::to_string(m_ResponsePort));

  m_LiveDataAcceptor = std::make_unique<asio::ip::tcp::acceptor>(
      m_Context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), m_LiveDataPort));
  logger.info("TCP Live Data server started on port " +
              std::to_string(m_LiveDataPort));

  m_IsServerRunning = true;
  while (m_IsServerRunning) {
    // Accept a new connection
    m_IsClientConnecting = false;
    if (!waitForNewConnexion()) {
      // If it failed to connect, restart the process
      continue;
    }

    // Start the command worker to handle the commands
    auto commandWorker = std::thread([this]() {
      while (m_IsServerRunning && isClientConnected()) {
        waitAndHandleNewCommand();
      }
    });

    auto responseWorker = std::thread([this]() {
      while (m_IsServerRunning && isClientConnected()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });

    auto liveDataWorker = std::thread([this]() {
      auto liveDataIntervals = std::chrono::milliseconds(100);
      std::this_thread::sleep_for(liveDataIntervals);
      while (m_IsServerRunning && isClientConnected()) {
        auto startingTime = std::chrono::high_resolution_clock::now();
        handleSendLiveData();
        auto next = liveDataIntervals -
                    (std::chrono::high_resolution_clock::now() - startingTime);
        std::this_thread::sleep_for(next);
      }
    });

    // Wait for the workers to finish
    commandWorker.join();
    responseWorker.join();
    liveDataWorker.join();
  }
}

void TcpServer::stopServer() {
  auto &logger = utils::Logger::getInstance();

  // Shutdown the server down
  if (m_IsServerRunning) {
    m_IsServerRunning = false;

    // When closing the server too soon, client may still be connected, so we
    // need to disconnect it
    disconnectClient();

    // Stop any running context
    m_Context.stop();
  }

  // Wait for the server to stop
  if (m_ServerWorker.joinable()) {
    m_ServerWorker.join();
  }
  logger.info("Server has shut down");
}

void TcpServer::disconnectClient() {
  std::lock_guard<std::mutex> lock(m_Mutex);
  auto &logger = utils::Logger::getInstance();

  logger.info("Disconnecting client");
  m_Status = TcpServerStatus::INITIALIZING;

  // Make sure all the devices are properly disconnected
  for (auto &name : m_Devices.getDeviceNames()) {
    removeDevice(name, false);
  }

  // Reset the status to initializing
  closeSockets();
}

bool TcpServer::isClientConnected() const {
  return m_Status != TcpServerStatus::INITIALIZING && m_CommandSocket &&
         m_CommandSocket->is_open() && m_ResponseSocket &&
         m_ResponseSocket->is_open() && m_LiveDataSocket &&
         m_LiveDataSocket->is_open();
}

bool TcpServer::waitForNewConnexion() {
  auto &logger = utils::Logger::getInstance();
  logger.info("Waiting for a new connexion");
  m_Status = TcpServerStatus::INITIALIZING;

  // Wait for the command socket to connect
  if (!waitUntilSocketIsConnected("Command", m_CommandSocket, m_CommandAcceptor,
                                  false)) {
    return false;
  }
  m_CommandSocket->non_blocking(true);

  // Wait for the response socket to connect
  logger.info("Command socket connected to client, waiting for a connexion to "
              "the response socket");
  if (!waitUntilSocketIsConnected("Response", m_ResponseSocket,
                                  m_ResponseAcceptor, true)) {
    return false;
  }

  // Wait for the live data socket to connect
  m_Status = TcpServerStatus::CONNECTING;
  logger.info("Response socket connected to client, waiting for a connexion to "
              "the live data socket");
  if (!waitUntilSocketIsConnected("LiveData", m_LiveDataSocket,
                                  m_LiveDataAcceptor, true)) {
    return false;
  }

  // Wait for the handshake
  logger.info("All ports are connected, waiting for the handshake");
  auto startingTime = std::chrono::high_resolution_clock::now();
  while (m_Status == TcpServerStatus::CONNECTING) {
    waitAndHandleNewCommand();

    // Since the command is non-blocking, we can just continue if there is no
    // data
    if (!m_IsServerRunning || !isClientConnected() ||
        std::chrono::high_resolution_clock::now() - startingTime >
            m_TimeoutPeriod) {
      logger.fatal("Handshake timeout (" +
                   std::to_string(m_TimeoutPeriod.count()) +
                   " ms), disconnecting client");
      disconnectClient();
      return false;
    }
  }
  return m_Status == TcpServerStatus::CONNECTED;
}

bool TcpServer::waitUntilSocketIsConnected(
    const std::string &socketName,
    std::unique_ptr<asio::ip::tcp::socket> &socket,
    std::unique_ptr<asio::ip::tcp::acceptor> &acceptor, bool canTimeout) {
  auto &logger = utils::Logger::getInstance();
  logger.info("Waiting for " + socketName + " socket to connect");

  socket = std::make_unique<asio::ip::tcp::socket>(m_Context);
  acceptor->async_accept(*socket, [](const asio::error_code &) {});

  auto startingTime = std::chrono::high_resolution_clock::now();
  auto timoutTimer =
      asio::steady_timer(m_Context, std::chrono::milliseconds(50));
  while (!socket->is_open()) {
    timoutTimer.async_wait(
        [this](const asio::error_code &) { m_Context.stop(); });
    m_Context.run();
    m_Context.restart();

    // Check for failing conditions
    if (!m_IsServerRunning) {
      logger.info("Stopping listening to ports as server is shutting down");
      return false;
    } else if (canTimeout &&
               std::chrono::high_resolution_clock::now() - startingTime >
                   m_TimeoutPeriod) {
      logger.fatal("Connexion to " + socketName + " socket timed out (" +
                   std::to_string(m_TimeoutPeriod.count()) +
                   " ms), disconnecting client");
      disconnectClient();
      return false;
    }
  }

  return true;
}

void TcpServer::closeSockets() {
  if (m_CommandSocket && m_CommandSocket->is_open()) {
    m_CommandSocket->close();
  }
  if (m_CommandAcceptor && m_CommandAcceptor->is_open()) {
    m_CommandAcceptor->cancel();
  }

  if (m_ResponseSocket && m_ResponseSocket->is_open()) {
    m_ResponseSocket->close();
  }
  if (m_ResponseAcceptor && m_ResponseAcceptor->is_open()) {
    m_ResponseAcceptor->cancel();
  }

  if (m_LiveDataSocket && m_LiveDataSocket->is_open()) {
    m_LiveDataSocket->close();
  }
  if (m_LiveDataAcceptor && m_LiveDataAcceptor->is_open()) {
    m_LiveDataAcceptor->cancel();
  }
}

void TcpServer::waitAndHandleNewCommand() {
  auto &logger = utils::Logger::getInstance();

  // Lock the mutex during the time the command is answered
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (!m_IsServerRunning || !isClientConnected()) {
      // Just a sanity check after the sleep and getting the lock
      return;
    }
  }

  auto buffer = std::array<char, BYTES_IN_CLIENT_PACKET_HEADER>();
  asio::error_code error;
  size_t byteRead = asio::read(*m_CommandSocket, asio::buffer(buffer), error);

  if (error == asio::error::eof) {
    logger.info("Client disconnected");
    disconnectClient();
    return;
  }

  // Since command is non-blocking, we can just continue if there is no data
  if (!m_IsServerRunning || byteRead == 0 ||
      error == asio::error::would_block) {
    return;
  }

  // If something went wrong, disconnect the client and stop everything
  if (byteRead > 1024 || error) {
    logger.fatal("TCP read error: " + error.message());
    disconnectClient();
    return;
  }

  // Parse the packet
  TcpServerCommand command = parseCommandPacket(buffer);

  // Handle the command based on the current status
  bool isSuccessful = false;
  switch (m_Status) {
  case (TcpServerStatus::CONNECTING):
    isSuccessful = handleHandshake(command);
    break;
  case (TcpServerStatus::CONNECTED):
    isSuccessful = handleCommand(command);
    break;
  default:
    logger.fatal("Invalid server status: " +
                 std::to_string(static_cast<std::uint32_t>(m_Status)));
    isSuccessful = false;
    break;
  }

  // If anything went wrong, disconnect the client
  if (!isSuccessful) {
    disconnectClient();
  }

  // If we get here, the command was successful and we can continue
  return;
}

bool TcpServer::handleHandshake(TcpServerCommand command) {
  auto &logger = utils::Logger::getInstance();
  asio::error_code error;

  // The only valid command during initialization is the handshake
  if (command != TcpServerCommand::HANDSHAKE) {
    logger.fatal("Invalid command during initialization: " +
                 std::to_string(static_cast<std::uint32_t>(command)));

    asio::write(*m_CommandSocket,
                asio::buffer(constructResponsePacket(TcpServerResponse::NOK)),
                error);
    return false;
  }

  // Respond OK to the handshake
  size_t byteWritten = asio::write(
      *m_CommandSocket,
      asio::buffer(constructResponsePacket(TcpServerResponse::OK)), error);
  if (byteWritten != BYTES_IN_SERVER_PACKET_HEADER || error) {
    logger.fatal("TCP write error: " + error.message());
    return false;
  }

  // Set the status to running
  m_Status = TcpServerStatus::CONNECTED;
  logger.info("Handshake from client is valid");

  return true;
}

bool TcpServer::handleCommand(TcpServerCommand command) {
  auto &logger = utils::Logger::getInstance();
  asio::error_code error;

  // Handle the command
  TcpServerResponse response;
  switch (command) {
  case TcpServerCommand::CONNECT_DELSYS_ANALOG:
    response = addDevice(DEVICE_NAME_DELSYS_ANALOG) ? TcpServerResponse::OK
                                                    : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::CONNECT_DELSYS_EMG:
    response = addDevice(DEVICE_NAME_DELSYS_EMG) ? TcpServerResponse::OK
                                                 : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::CONNECT_MAGSTIM:
    response = addDevice(DEVICE_NAME_MAGSTIM) ? TcpServerResponse::OK
                                              : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::ZERO_DELSYS_ANALOG:
    response = m_Devices.zeroLevelDevice(DEVICE_NAME_DELSYS_ANALOG)
                   ? TcpServerResponse::OK
                   : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::ZERO_DELSYS_EMG:
    response = m_Devices.zeroLevelDevice(DEVICE_NAME_DELSYS_EMG)
                   ? TcpServerResponse::OK
                   : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::DISCONNECT_DELSYS_ANALOG:
    response = removeDevice(DEVICE_NAME_DELSYS_ANALOG) ? TcpServerResponse::OK
                                                       : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::DISCONNECT_DELSYS_EMG:
    response = removeDevice(DEVICE_NAME_DELSYS_EMG) ? TcpServerResponse::OK
                                                    : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::DISCONNECT_MAGSTIM:
    response = removeDevice(DEVICE_NAME_MAGSTIM) ? TcpServerResponse::OK
                                                 : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::START_RECORDING:
    response = m_Devices.startRecording() ? TcpServerResponse::OK
                                          : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::STOP_RECORDING:
    response = m_Devices.stopRecording() ? TcpServerResponse::OK
                                         : TcpServerResponse::NOK;
    break;

  case TcpServerCommand::GET_LAST_TRIAL_DATA: {
    auto data = m_Devices.getLastTrialDataSerialized();
    auto dataDump = data.dump();
    asio::write(*m_ResponseSocket,
                asio::buffer(constructResponsePacket(
                    static_cast<TcpServerResponse>(dataDump.size()))),
                error);
    auto written =
        asio::write(*m_ResponseSocket, asio::buffer(dataDump), error);
    logger.info("Data size: " + std::to_string(written));
    response = TcpServerResponse::OK;
  } break;

  default:
    logger.fatal("Invalid command: " +
                 std::to_string(static_cast<std::uint32_t>(command)));
    response = TcpServerResponse::NOK;
    break;
  }

  // Respond OK to the command
  size_t byteWritten = asio::write(
      *m_CommandSocket, asio::buffer(constructResponsePacket(response)), error);
  if (byteWritten != BYTES_IN_SERVER_PACKET_HEADER || error) {
    logger.fatal("TCP write error: " + error.message());
    return false;
  }

  return true;
}

TcpServerCommand TcpServer::parseCommandPacket(
    const std::array<char, BYTES_IN_CLIENT_PACKET_HEADER> &buffer) {
  // Packets are exactly 8 bytes long, big-endian
  // - First 4 bytes are the version number
  // - Next 4 bytes are the command

  // Check the version
  std::uint32_t version =
      *reinterpret_cast<const std::uint32_t *>(buffer.data());
  if (version != m_ProtocolVersion) {
    auto &logger = utils::Logger::getInstance();
    logger.fatal("Invalid version: " + std::to_string(version) +
                 ". Please "
                 "update the client to version " +
                 std::to_string(m_ProtocolVersion));
    return TcpServerCommand::FAILED;
  }

  // Get the command
  return static_cast<TcpServerCommand>(
      *reinterpret_cast<const std::uint32_t *>(buffer.data() + 4));
}

std::array<char, BYTES_IN_SERVER_PACKET_HEADER>
TcpServer::constructResponsePacket(TcpServerResponse response) {
  // Packets are exactly 16 bytes long, big-endian
  // - First 4 bytes are the version number
  // - The next 8 bytes are the timestamp of the packet (milliseconds since
  // epoch)
  // - Next 4 bytes are the actual response

  auto packet = std::array<char, BYTES_IN_SERVER_PACKET_HEADER>();
  packet.fill('\0');

  // Add the version number
  std::memcpy(packet.data(), &m_ProtocolVersion, sizeof(std::uint32_t));

  // Add the timestamps in uint64_t format
  std::uint64_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  std::memcpy(packet.data() + sizeof(std::uint32_t), &timestamp,
              sizeof(std::uint64_t));

  // Add the response
  std::memcpy(packet.data() + sizeof(std::uint32_t) + sizeof(std::uint64_t),
              &response, sizeof(TcpServerResponse));

  return packet;
}

bool TcpServer::addDevice(const std::string &deviceName) {
  auto &logger = utils::Logger::getInstance();

  // Check if [m_ConnectedDeviceIds] contains the device
  if (m_ConnectedDeviceIds.find(deviceName) != m_ConnectedDeviceIds.end()) {
    logger.warning("Cannot add the " + deviceName +
                   " devise as it is already connected");
    return false;
  }

  // Stop the data streaming when changing the devices
  m_Devices.stopDataStreaming();
  makeAndAddDevice(deviceName);
  if (!m_Devices.connect()) {
    removeDevice(deviceName);
    return false;
  }
  m_Devices.startDataStreaming();

  return true;
}

bool TcpServer::setZeroLevel(const std::string &deviceName) {
  auto &logger = utils::Logger::getInstance();

  // Check if [m_ConnectedDeviceIds] contains the device
  if (m_ConnectedDeviceIds.find(deviceName) == m_ConnectedDeviceIds.end()) {
    logger.warning(deviceName + " not connected");
    return false;
  }

  m_Devices.zeroLevelDevice(deviceName);
  return true;
}

void TcpServer::makeAndAddDevice(const std::string &deviceName) {
  auto &logger = utils::Logger::getInstance();

  if (deviceName == DEVICE_NAME_DELSYS_ANALOG) {
    bool isInitialized = false;
    for (auto &id : m_Devices.getDeviceIds()) {
      if (dynamic_cast<const devices::DelsysBaseDevice *>(&m_Devices[id])) {
        isInitialized = true;
        m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_ANALOG] =
            m_Devices.add(std::make_unique<devices::DelsysAnalogDevice>(
                static_cast<const devices::DelsysBaseDevice &>(m_Devices[id])));
        break;
      }
    }
    if (!isInitialized) {
      m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_ANALOG] =
          m_Devices.add(std::make_unique<devices::DelsysAnalogDevice>());
    }

  } else if (deviceName == DEVICE_NAME_DELSYS_EMG) {
    bool isInitialized = false;
    for (auto &id : m_Devices.getDeviceIds()) {
      if (dynamic_cast<const devices::DelsysBaseDevice *>(&m_Devices[id])) {
        isInitialized = true;
        m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_EMG] =
            m_Devices.add(std::make_unique<devices::DelsysEmgDevice>(
                static_cast<const devices::DelsysBaseDevice &>(m_Devices[id])));
        break;
      }
    }
    if (!isInitialized) {
      m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_EMG] =
          m_Devices.add(std::make_unique<devices::DelsysEmgDevice>());
    }

  } else if (deviceName == DEVICE_NAME_MAGSTIM) {
    m_ConnectedDeviceIds[DEVICE_NAME_MAGSTIM] =
        m_Devices.add(devices::MagstimRapidDevice::findMagstimDevice());

  } else {
    logger.fatal("Invalid device name: " + deviceName);
    throw std::runtime_error("Invalid device name: " + deviceName);
  }
}

bool TcpServer::removeDevice(const std::string &deviceName,
                             bool restartStreaming) {
  auto &logger = utils::Logger::getInstance();

  // Check if [m_ConnectedDeviceIds] contains the device
  if (m_ConnectedDeviceIds.find(deviceName) == m_ConnectedDeviceIds.end()) {
    logger.warning(deviceName + " not connected");
    return false;
  }

  // Stop the data streaming when changing the devices
  m_Devices.stopDataStreaming();
  m_Devices.remove(m_ConnectedDeviceIds[deviceName]);
  m_ConnectedDeviceIds.erase(deviceName);
  if (restartStreaming) {
    m_Devices.startDataStreaming();
  }

  return true;
}

void TcpServer::handleSendLiveData() {
  // Send the live data callback in a separate thread
  if (!isClientConnected()) {
    return;
  }
  auto &logger = utils::Logger::getInstance();
  logger.debug("Sending live data to client");

  auto data = m_Devices.getLiveDataSerialized();
  if (data.size() == 0) {
    return;
  }

  auto dataDump = data.dump();
  asio::error_code error;
  asio::write(*m_LiveDataSocket,
              asio::buffer(constructResponsePacket(
                  static_cast<TcpServerResponse>(dataDump.size()))),
              error);
  auto written = asio::write(*m_LiveDataSocket, asio::buffer(dataDump), error);
  logger.debug("Live data size: " + std::to_string(written));
}

void TcpServerMock::makeAndAddDevice(const std::string &deviceName) {
  auto &logger = utils::Logger::getInstance();

  if (deviceName == DEVICE_NAME_DELSYS_ANALOG) {
    bool isInitialized = false;
    for (auto &id : m_Devices.getDeviceIds()) {
      if (dynamic_cast<const devices::DelsysBaseDevice *>(&m_Devices[id])) {
        isInitialized = true;
        m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_ANALOG] =
            m_Devices.add(std::make_unique<devices::DelsysAnalogDeviceMock>(
                static_cast<const devices::DelsysBaseDevice &>(m_Devices[id])));
        break;
      }
    }
    if (!isInitialized) {
      m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_ANALOG] =
          m_Devices.add(std::make_unique<devices::DelsysAnalogDeviceMock>());
    }

  } else if (deviceName == DEVICE_NAME_DELSYS_EMG) {
    bool isInitialized = false;
    for (auto &id : m_Devices.getDeviceIds()) {
      if (dynamic_cast<const devices::DelsysBaseDevice *>(&m_Devices[id])) {
        isInitialized = true;
        m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_EMG] =
            m_Devices.add(std::make_unique<devices::DelsysEmgDeviceMock>(
                static_cast<const devices::DelsysBaseDevice &>(m_Devices[id])));
        break;
      }
    }
    if (!isInitialized) {
      m_ConnectedDeviceIds[DEVICE_NAME_DELSYS_EMG] =
          m_Devices.add(std::make_unique<devices::DelsysEmgDeviceMock>());
    }

  } else if (deviceName == DEVICE_NAME_MAGSTIM) {
    m_ConnectedDeviceIds[DEVICE_NAME_MAGSTIM] =
        m_Devices.add(devices::MagstimRapidDeviceMock::findMagstimDevice());

  } else {
    logger.fatal("Invalid device name: " + deviceName);
    throw std::runtime_error("Invalid device name: " + deviceName);
  }
}