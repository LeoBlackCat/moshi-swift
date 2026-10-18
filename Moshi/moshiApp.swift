// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import SwiftUI

func requestMicrophoneAccess() {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return
    case .notDetermined:
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            print("granted", granted)
        }
    case .denied:  // The user has previously denied access.
        return
    case .restricted:  // The user can't grant access due to restrictions.
        return
    case _:
        return
    }
}

@main
struct moshiApp: App {
    @Environment(\.scenePhase) var scenePhase

    init() {
        // MLX reads this once on first Metal evaluation: configure before constructing models.
        let args = ProcessInfo.processInfo.arguments
        var operations = 64
        if let index = args.firstIndex(of: "--metal-ops"), args.count > index + 1,
           let requested = Int(args[index + 1]), [10, 32, 64, 128].contains(requested) {
            operations = requested
        }
        setenv("MLX_MAX_OPS_PER_BUFFER", String(operations), 1)
        requestMicrophoneAccess()
    }

    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.arguments.contains("--benchmark") {
                NavigationStack { BenchmarkView() }
            } else {
                ContentView().environment(DeviceStat())
            }
        }
#if os(iOS)
        .onChange(of: scenePhase) { (phase) in
            switch phase {
            case .active:
                // In iOS 13+, idle timer needs to be set in scene to override default
                UIApplication.shared.isIdleTimerDisabled = true
            case .inactive: break
            case .background: break
            @unknown default: print("ScenePhase: unexpected state")
            }
        }
#endif
    }
}
