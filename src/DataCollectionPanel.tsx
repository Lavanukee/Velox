import React from 'react';
import { Construction } from 'lucide-react';
import './styles/legacy.css';

interface DataCollectionPanelProps {
  addLogMessage: (message: string) => void;
}

const DataCollectionPanel: React.FC<DataCollectionPanelProps> = ({ addLogMessage }) => {
  return (
    <div className="data-collection-container" style={{ position: 'relative', minHeight: '600px' }}>
      {/* Blurred Content */}
      <div style={{ filter: 'blur(4px)', pointerEvents: 'none', opacity: 0.4 }}>
        <div className="section-header">
          <h1>Collect Your Own Data</h1>
          <p style={{ color: 'var(--text-muted)' }}>Create custom datasets for your specific use cases</p>
        </div>

        <div className="collection-options">
          <div className="collection-card featured">
            <div className="collection-icon">🤖</div>
            <h3>Computer Agent Dataset</h3>
            <p className="collection-description">
              Create a dataset to make a computer agent proficient across the apps you use.
              The annotation tool serves as a base for capturing UI interactions and building
              training data for vision-language models.
            </p>

            <div className="collection-steps">
              <div className="step">
                <span className="step-number">1</span>
                <span>Capture screenshots of your application workflows</span>
              </div>
              <div className="step">
                <span className="step-number">2</span>
                <span>Annotate UI elements and actions using the annotation tool</span>
              </div>
              <div className="step">
                <span className="step-number">3</span>
                <span>Generate training pairs of images and instructions</span>
              </div>
              <div className="step">
                <span className="step-number">4</span>
                <span>Fine-tune your model on the collected dataset</span>
              </div>
            </div>

            <button
              className="btn-primary btn-large"
              onClick={() => {
                addLogMessage('Opening annotation tool...');
              }}
            >
              Launch Annotation Tool
            </button>

            <div className="coming-soon-badge">
              <p style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                📋 More data collection methods coming soon:
              </p>
              <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginLeft: '1.5rem' }}>
                <li>Web scraping pipelines</li>
                <li>Multi-modal data capture</li>
                <li>Synthetic data generation</li>
                <li>Active learning workflows</li>
              </ul>
            </div>
          </div>

          <div className="collection-card placeholder">
            <div className="collection-icon">🌐</div>
            <h3>Web Data Collection</h3>
            <p className="collection-description">
              Automated web scraping and data extraction workflows.
            </p>
            <div className="placeholder-overlay">
              <span>Coming Soon</span>
            </div>
          </div>

          <div className="collection-card placeholder">
            <div className="collection-icon">🎨</div>
            <h3>Synthetic Data</h3>
            <p className="collection-description">
              Generate synthetic training data using existing models
            </p>
            <div className="placeholder-overlay">
              <span>Coming Soon</span>
            </div>
          </div>

          <div className="collection-card placeholder">
            <div className="collection-icon">🔄</div>
            <h3>Active Learning</h3>
            <p className="collection-description">
              Iteratively improve your dataset with model-in-the-loop selection.
            </p>
            <div className="placeholder-overlay">
              <span>Coming Soon</span>
            </div>
          </div>
        </div>
      </div>

      {/* Under Construction Overlay */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        textAlign: 'center',
        zIndex: 10,
        background: 'var(--bg-overlay)',
        backdropFilter: 'blur(12px)',
        padding: '3rem 4rem',
        borderRadius: 'var(--radius-lg)',
        border: '2px solid var(--accent-primary)',
        boxShadow: 'var(--shadow-lg)',
      }}>
        <Construction size={64} style={{
          color: 'var(--accent-primary)',
          marginBottom: '1.5rem',
          animation: 'pulse 2s ease-in-out infinite'
        }} />
        <h2 style={{
          fontSize: '2rem',
          fontWeight: 700,
          marginBottom: '0.5rem',
          background: 'var(--accent-gradient)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          color: 'transparent'
        }}>
          Under Construction
        </h2>
        <p style={{
          fontSize: '1.1rem',
          color: 'var(--text-secondary)',
          marginBottom: '1rem',
          fontWeight: 600
        }}>
          (WIP)
        </p>
        <p style={{
          fontSize: '0.95rem',
          color: 'var(--text-main)',
          maxWidth: '400px'
        }}>
          Collect your own data easily in a variety of ways.
        </p>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
};

export default DataCollectionPanel;