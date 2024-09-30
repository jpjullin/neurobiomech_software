#ifndef __STIMWALKER_DEVICES_NI_DAQ_DEVICE_H__
#define __STIMWALKER_DEVICES_NI_DAQ_DEVICE_H__

#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "Devices/Generic/DataCollector.h"
#include "Devices/Generic/Device.h"
#include "stimwalkerConfig.h"

namespace STIMWALKER_NAMESPACE::devices {

/// @brief Abstract class for devices
class NidaqDevice : public Device, public DataCollector {

public:
  /// @brief Constructor
  /// @param channelCount The number of channels
  /// @param frameRate The frame rate
  NidaqDevice(size_t channelCount, size_t frameRate);

  // Delete copy constructor and assignment operator, this class cannot be
  // copied because of the mutex member
  NidaqDevice(const NidaqDevice &) = delete;
  NidaqDevice &operator=(const NidaqDevice &) = delete;

  ~NidaqDevice();

  void connect() override;
  void disconnect() override;
  void startRecording() override;
  void stopRecording() override;

protected:
  /// @brief Notify the listeners that new data has been collected
  void HandleNewData(const data::DataPoint &data) override;
};

// // TODO Reimplement the mocker
// class NidaqDeviceMock : public NidaqDevice {
// public:
//   NidaqDeviceMock(int channelCount, int frameRate);

//   // Delete copy constructor and assignment operator, this class cannot be
//   // copied because of the mutex member
//   NidaqDeviceMock(const NidaqDeviceMock &) = delete;
//   NidaqDeviceMock &operator=(const NidaqDeviceMock &) = delete;

//   void startRecording() override;
//   void stopRecording() override;

// protected:
//   /// @brief Simulate the recording
//   void generateData();

//   ///< Should the mock continue generating data (thread safe)
//   bool m_generateData = false;
// };

} // namespace STIMWALKER_NAMESPACE::devices

#endif // __STIMWALKER_DEVICES_NI_DAQ_DEVICE_H__