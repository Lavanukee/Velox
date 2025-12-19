import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AppProvider, useApp } from "./context/AppContext";
import { AppStateProvider } from "./context/AppStateContext";
import { Layout } from "./components/Layout";
import { AppView, ModelConfig, DownloadTask } from "./types";
import { ErrorBoundary } from "./components/ErrorBoundary";


// Pages
import FineTuningPage from "./pages/FineTuningPage";
import InferencePage from "./pages/InferencePage";
import SettingsPage from "./pages/SettingsPage";

// Legacy Components
import ResourceDashboard from "./ResourceDashboardNew";
import UtilitiesPanel from "./UtilitiesPanel";
import DataCollectionPanel from "./DataCollectionPanel";

// Styles
import "./styles/global.css";

interface Notification {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
}

function AppContent() {
  const [currentView, setCurrentView] = useState(AppView.Dashboard);
  const [logs, setLogs] = useState<string[]>([]);
  const [selectedModelConfig, setSelectedModelConfig] = useState<ModelConfig | null>(null);

  // Download management
  const [downloadTasks, setDownloadTasks] = useState<DownloadTask[]>([]);
  const [notifications, setNotifications] = useState<Notification[]>([]);

  // Python Setup State (from Context)
  const {
    showPythonSetup, setShowPythonSetup,
    isInitializing, setIsInitializing,
    setupProgressPercent, setSetupProgressPercent,
    setupMessage, setSetupMessage,
    setupLoadedBytes, setSetupLoadedBytes,
    setupTotalBytes, setSetupTotalBytes,
    runGlobalSetup
  } = useApp();

  const updateDownloadTask = useCallback((id: string, update: Partial<DownloadTask>) => {
    setDownloadTasks(prev => {
      const taskIndex = prev.findIndex(task => task.id === id);

      if (taskIndex === -1) {
        console.warn(`Attempted to update unknown task ID: ${id}`);
        return prev;
      }

      const updatedTask = { ...prev[taskIndex], ...update };
      let newTasks = prev.map((task, index) => index === taskIndex ? updatedTask : task);

      if (updatedTask.status === 'completed' || updatedTask.status === 'error' || updatedTask.status === 'cancelled') {
        setTimeout(() => {
          setDownloadTasks(current => current.filter(task => task.id !== id));
        }, 8000);
      }

      return newTasks;
    });
  }, []);

  const addLogMessage = (message: string) => {
    setLogs((prevLogs) => {
      if (prevLogs.length > 0 && prevLogs[prevLogs.length - 1] === message) {
        return prevLogs;
      }
      const newLogs = [...prevLogs, message];
      console.log(`APP_LOG (UI): ${message}`);
      return newLogs;
    });
  };

  const addNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const id = `notif_${Date.now()}`;
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  useEffect(() => {
    let isMounted = true;
    // No more hardcoded 12s timeout. We rely on checkEnvironment
    // to finish and trigger the SplashOverlay completion.

    const unlistenLog = listen("log", (event) => {
      if (isMounted) addLogMessage(`TAURI_BACKEND: ${event.payload as string}`);
    });

    const unlistenProgress = listen<{ id: string, progress: number }>("download_progress", (event) => {
      if (isMounted) updateDownloadTask(event.payload.id, { progress: event.payload.progress, status: 'downloading' });
    });

    const unlistenStatus = listen<{ id: string, status: DownloadTask['status'] }>("download_status", (event) => {
      const { id, status } = event.payload;
      if (isMounted) {
        if (status === 'completed') {
          updateDownloadTask(id, { progress: 100, status: 'completed' });
        } else if (status === 'error' || status === 'cancelled') {
          updateDownloadTask(id, { status });
        }
      }
    });

    const unlistenSetup = listen<{ step: string, message: string, progress: number, loaded?: number, total?: number }>("setup_progress", (event) => {
      if (isMounted) {
        setSetupProgressPercent(event.payload.progress);
        if (event.payload.message) setSetupMessage(event.payload.message);
        if (event.payload.loaded !== undefined) setSetupLoadedBytes(event.payload.loaded);
        if (event.payload.total !== undefined) setSetupTotalBytes(event.payload.total);
      }
    });

    const initializeApp = async () => {
      try {
        const configs: ModelConfig[] = await invoke("load_model_configs_command");
        if (isMounted && configs.length > 0) {
          setSelectedModelConfig(configs[0]);
        }
      } catch (error) {
        if (isMounted) addLogMessage(`ERROR loading configs: ${error}`);
      }
    };

    const checkEnvironment = async () => {
      try {
        if (isMounted) setSetupMessage("Verifying Environment...");

        // 1. Check Python Dependencies
        const isPythonReady = await invoke<boolean>('check_python_installed_command');
        if (!isPythonReady && isMounted) {
          setSetupMessage("Python Environment incomplete. Auto-repairing...");
          await runGlobalSetup(true);
          if (isMounted) addNotification("Python Environment Repaired!", "success");
        }

        // 2. Check Llama Binary & Updates
        if (isMounted) setSetupMessage("Checking Inference Engine...");
        const isLlamaReady = await invoke<boolean>('check_llama_binary_command');
        if (!isLlamaReady && isMounted) {
          setSetupMessage("Updating Inference Engine...");
          setSetupProgressPercent(0);
          await invoke('download_llama_binary_command');
          if (isMounted) {
            setSetupProgressPercent(100);
            addNotification("Inference Engine Updated!", "success");
          }
        }

        if (isMounted) {
          setSetupProgressPercent(100);
          setSetupMessage("SYSTEM_READY");
        }

      } catch (error) {
        if (isMounted) {
          addLogMessage(`Environment check error: ${error}`);
          addNotification(`Startup Setup Failed: ${error}`, "error");
          // Even on error, we should eventually allow the user into the app
          setSetupProgressPercent(100);
          setSetupMessage("SYSTEM_READY");
        }
      } finally {
        if (isMounted) {
          // We don't hide immediately here, we let the SplashOverlay 
          // animation finish and call onSplashComplete.
        }
      }
    };

    initializeApp();
    checkEnvironment();

    return () => {
      isMounted = false;
      unlistenLog.then((f) => f());
      unlistenProgress.then(f => f());
      unlistenStatus.then(f => f());
      unlistenSetup.then(f => f());
    };
  }, [updateDownloadTask, runGlobalSetup]);

  const getTitle = () => {
    switch (currentView) {
      case AppView.Dashboard: return 'Dashboard';
      case AppView.Utilities: return 'Utilities';
      case AppView.DataCollection: return 'Data Collection';
      case AppView.FineTuning: return 'Fine-Tuning Studio';
      case AppView.Inference: return 'Inference Chat';
      case AppView.Logs: return 'System Logs';
      case AppView.Settings: return 'Settings';
      default: return '';
    }
  };

  return (
    <Layout
      currentView={currentView}
      onNavigate={setCurrentView}
      title={getTitle()}
      downloadTasks={downloadTasks}
      isSettingUp={showPythonSetup || isInitializing}
      setupProgress={setupProgressPercent}
      setupMessage={setupMessage}
      setupLoadedBytes={setupLoadedBytes}
      setupTotalBytes={setupTotalBytes}
      onSplashComplete={() => {
        setIsInitializing(false);
        setShowPythonSetup(false);
      }}
    >
      {/* Global Notifications - Premium styled */}
      <div style={{ position: 'fixed', top: '96px', right: '24px', zIndex: 100, display: 'flex', flexDirection: 'column', gap: '12px', pointerEvents: 'none' }}>
        {notifications.map(notif => (
          <div
            key={notif.id}
            style={{
              pointerEvents: 'auto',
              background: notif.type === 'success'
                ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.95) 0%, rgba(5, 150, 105, 0.95) 100%)'
                : notif.type === 'error'
                  ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.95) 0%, rgba(220, 38, 38, 0.95) 100%)'
                  : 'linear-gradient(135deg, rgba(59, 130, 246, 0.95) 0%, rgba(37, 99, 235, 0.95) 100%)',
              padding: '16px 20px',
              borderRadius: '12px',
              backdropFilter: 'blur(12px)',
              border: notif.type === 'success'
                ? '1px solid rgba(16, 185, 129, 0.3)'
                : notif.type === 'error'
                  ? '1px solid rgba(239, 68, 68, 0.3)'
                  : '1px solid rgba(59, 130, 246, 0.3)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.2)',
              minWidth: '320px',
              maxWidth: '420px',
              animation: 'slideInRight 0.3s ease-out'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'rgba(255, 255, 255, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}>
                {notif.type === 'success' && (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3,8 6,11 13,4" />
                  </svg>
                )}
                {notif.type === 'error' && (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="4" y1="4" x2="12" y2="12" />
                    <line x1="12" y1="4" x2="4" y2="12" />
                  </svg>
                )}
                {notif.type === 'info' && (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="8" cy="8" r="6" />
                    <line x1="8" y1="7" x2="8" y2="11" />
                    <circle cx="8" cy="5" r="0.5" fill="white" />
                  </svg>
                )}
              </div>
              <div style={{ flex: 1, color: 'white', fontSize: '14px', fontWeight: 500, lineHeight: '1.4' }}>
                {notif.message}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Content */}
      {currentView === AppView.Dashboard && (
        <ErrorBoundary>
          <ResourceDashboard
            addLogMessage={addLogMessage}
            addNotification={addNotification}
            setDownloadTasks={setDownloadTasks}
          />
        </ErrorBoundary>
      )}

      {currentView === AppView.Utilities && (
        <UtilitiesPanel
          addLogMessage={addLogMessage}
          addNotification={addNotification}
        />
      )}

      {currentView === AppView.DataCollection && (
        <DataCollectionPanel addLogMessage={addLogMessage} />
      )}

      {currentView === AppView.FineTuning && (
        <FineTuningPage
          addLogMessage={addLogMessage}
          addNotification={addNotification}
        />
      )}

      {currentView === AppView.Inference && (
        <InferencePage
          modelConfig={selectedModelConfig}
          addLogMessage={addLogMessage}
        />
      )}

      {currentView === AppView.Settings && (
        <SettingsPage
          onReinstallPython={async () => {
            try {
              await runGlobalSetup(true);
              addNotification("Reinstall complete", "success");
            } catch (e) {
              addNotification(`Reinstall failed: ${e}`, "error");
            }
          }}
          onReinstallDependencies={async () => {
            try {
              await runGlobalSetup(false);
              addNotification("Dependencies updated", "success");
            } catch (e) {
              addNotification(`Update failed: ${e}`, "error");
            }
          }}
        />
      )}

      {currentView === AppView.Logs && (
        <div className="bg-panel rounded-xl border border-white/10 p-6 font-mono text-sm h-[calc(100vh-140px)] overflow-y-auto" style={{ background: '#121216', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', padding: '24px', fontFamily: 'monospace', height: 'calc(100vh - 140px)', overflowY: 'auto' }}>
          {logs.length === 0 ? <div className="text-gray-500">No logs yet...</div> : logs.map((log, index) => (
            <div key={index} className="mb-1 text-gray-300 border-b border-white/5 pb-1">
              <span className="text-gray-500 mr-2">[{index}]</span>
              {log}
            </div>
          ))}
          <button
            className="mt-4 px-3 py-1 bg-white/5 hover:bg-white/10 rounded text-xs text-gray-400"
            onClick={() => setLogs([])}
          >
            Clear Logs
          </button>
        </div>
      )}
    </Layout>
  );
}

function App() {
  return (
    <AppStateProvider>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </AppStateProvider>
  );
}

export default App;