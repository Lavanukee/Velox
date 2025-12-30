import { useState, useEffect, useCallback, useRef } from "react";
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
    runGlobalSetup,
    setIsEngineUpdating
  } = useApp();


  const isBackgroundUpdateRef = useRef(false);

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
      if (prevLogs.length > 0 && prevLogs.length < 1000 && prevLogs[prevLogs.length - 1] === message) {
        return prevLogs;
      }
      // Cap logs to prevent memory leak
      const newLogs = prevLogs.length > 1000 ? [...prevLogs.slice(100), message] : [...prevLogs, message];
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
      if (isMounted && !isBackgroundUpdateRef.current) {
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
        // 1. Check Python Dependencies (Fast Path)
        // We use a minimal check to get the user in quickly.
        const isPythonMinimalReady = await invoke<boolean>('check_python_minimal_command');

        if (!isPythonMinimalReady && isMounted) {
          setSetupMessage("Python Environment incomplete. Auto-repairing...");
          await runGlobalSetup(true);
          if (isMounted) addNotification("Python Environment Repaired!", "success");
        }

        // If minimal is ready, we let them through to the next check, 
        // but we kick off a full verification in background? 
        // Actually, if we want to be safe, we just trust minimal for now.
        // If full check fails later, they will get errors when trying to run things.

        /*
        const isPythonReady = await invoke<boolean>('check_python_installed_command');
        if (!isPythonReady && isMounted) {
          setSetupMessage("Python Environment incomplete. Auto-repairing...");
          await runGlobalSetup(true);
          if (isMounted) addNotification("Python Environment Repaired!", "success");
        }
        */

        // 2. Check Llama Binary & Updates
        if (isMounted) setSetupMessage("Checking Inference Engine...");
        const isLlamaReady = await invoke<boolean>('check_llama_binary_command');

        if (!isLlamaReady && isMounted) {
          // Check if binary actually exists OR if it just needs update
          const binExists = await invoke<boolean>('check_llama_binary_exists_command');

          if (!binExists) {
            setSetupMessage("Installing Inference Engine...");
            setSetupProgressPercent(0);
            await invoke('download_llama_binary_command');
            if (isMounted) setSetupProgressPercent(100);
          } else {
            // It exists but needs update (version mismatch)
            // WE DISMISS SPLASH EARLY HERE
            setSetupMessage("SYSTEM_READY");
            setSetupProgressPercent(100);

            // Trigger background update
            setIsEngineUpdating(true);
            isBackgroundUpdateRef.current = true;

            invoke('download_llama_binary_command').then(() => {
              if (isMounted) {
                setIsEngineUpdating(false);
                isBackgroundUpdateRef.current = false;
                addNotification("Inference Engine Updated!", "success");
              }
            }).catch(e => {
              if (isMounted) {
                setIsEngineUpdating(false);
                isBackgroundUpdateRef.current = false;
                addNotification(`Engine Update Failed: ${e}`, "error");
              }
            });

            return; // Exit checkEnvironment early as we are ready to open
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
              background: 'rgba(18, 18, 22, 0.95)',
              backdropFilter: 'blur(12px)',
              border: `1px solid ${notif.type === 'error' ? 'rgba(239, 68, 68, 0.3)' : notif.type === 'success' ? 'rgba(34, 197, 94, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
              borderRadius: '16px',
              padding: '16px 20px',
              boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              maxWidth: '400px',
              animation: 'slideInRight 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
            }}
          >
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: notif.type === 'error' ? 'rgba(239, 68, 68, 0.15)' : notif.type === 'success' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 255, 255, 0.05)',
              color: notif.type === 'error' ? '#f87171' : notif.type === 'success' ? '#4ade80' : '#fff'
            }}>
              {notif.type === 'error' ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
              )}
            </div>
            <div style={{ flex: 1, color: 'white', fontSize: '14px', fontWeight: 500, lineHeight: '1.4' }}>
              {notif.message}
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
        <div className="logs-container">
          <div className="logs-header">
            <h3>System Logs</h3>
            <button
              className="btn btn-sm btn-ghost"
              onClick={() => setLogs([])}
            >
              Clear Logs
            </button>
          </div>
          <div className="logs-content">
            {logs.length === 0 ? (
              <div className="text-dim" style={{ padding: '20px', textAlign: 'center' }}>No logs yet...</div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="log-entry">
                  <span className="log-index">[{index}]</span>
                  <span className="log-message">{log}</span>
                </div>
              ))
            )}
          </div>
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