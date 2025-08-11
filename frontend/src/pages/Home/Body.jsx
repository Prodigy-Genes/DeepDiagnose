import React, { useState, useEffect } from 'react';
import { Upload, Zap, Shield, Users, Brain, Activity, FileText, Code, ArrowRight, Check } from 'lucide-react';
import SignIn from '../../auth/Sign-In/SignIn';
import SignUp from '../../auth/Sign-Up/SignUp';
import './Body.css';

const Body = () => {
    const [activeModel, setActiveModel] = useState('pneumonia');
    const [isVisible, setIsVisible] = useState(false);
    const [showAuth, setShowAuth] = useState(false);
    const [authMode, setAuthMode] = useState('signup'); // 'signin' or 'signup'

    useEffect(() => {
        setIsVisible(true);
    }, []);

    const handleClick = () => {
        // Check if user is authenticated
        const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
        
        if (token) {
            // User is authenticated, redirect to upload page
            const uploadUrl = `${window.location.origin}/upload`;
            window.open(uploadUrl, '_blank', 'noopener,noreferrer');
        } else {
            // User is not authenticated, show signup modal
            setAuthMode('signup');
            setShowAuth(true);
        }
    };

    const handleLearnMore = () => {
        // Redirect to the about page
        window.location.href = `${window.location.origin}/about`;
    };

    const handleToggleAuth = () => {
        setAuthMode(authMode === 'signin' ? 'signup' : 'signin');
    };

    const handleCloseAuth = () => {
        setShowAuth(false);
    };

    const models = {
        pneumonia: {
            name: 'Pneumonia Detection',
            type: 'Chest X-Ray',
            accuracy: '94.2%',
            description: 'Advanced AI model trained to detect pneumonia patterns in chest X-ray images with high precision.',
            features: ['Instant analysis', 'Confidence scoring', 'Expert summary']
        },
        osteoarthritis: {
            name: 'Osteoarthritis Detection',
            type: 'Knee X-Ray',
            accuracy: '91.8%',
            description: 'Specialized model for identifying osteoarthritis severity and joint degradation in knee X-rays.',
            features: ['Severity grading', 'Joint assessment', 'Treatment insights']
        },
        covid: {
            name: 'COVID-19 Detection',
            type: 'CT Scan',
            accuracy: '96.1%',
            description: 'State-of-the-art model for detecting COVID-19 related lung abnormalities in CT scans.',
            features: ['Lung pattern analysis', 'Infection mapping', 'Risk assessment']
        }
    };

    const stats = [
        { number: '3', label: 'AI Disease Models', icon: Brain },
        { number: '94%', label: 'Average Accuracy', icon: Activity },
        { number: '100%', label: 'Free Access', icon: Shield },
        { number: '24/7', label: 'Availability', icon: Zap }
    ];

    return (
        <>
            <div className={`b-medical-ai-body ${isVisible ? 'b-fade-in' : ''}`}>
                {/* Animated Background Elements */}
                <div className="b-background-elements">
                    <div className="b-dna-helix"></div>
                    <div className="b-pulse-rings"></div>
                    <div className="b-floating-particles"></div>
                </div>

                {/* Hero Section */}
                <section className="b-hero-section">
                    <div className="b-hero-container">
                        <div className="b-hero-content">
                            <div className="b-hero-badge">
                                <span className="b-badge-text">Revolutionary AI Technology</span>
                            </div>
                            <h1 className="b-hero-title">
                                <span className="b-title-gradient">Medical Imaging</span>
                                <span className="b-title-accent">Redefined</span>
                            </h1>
                            <p className="b-hero-description">
                                Democratizing medical diagnostics with cutting-edge AI technology. 
                                Get instant, accurate analysis of your medical scans - completely free for everyone.
                            </p>
                            <div className="b-hero-actions">
                                <button className="b-cta-primary" onClick={handleClick}>
                                    <span>Start Analysis</span>
                                    <ArrowRight className="b-cta-icon" />
                                </button>
                                <button className="b-cta-secondary" onClick={handleLearnMore}>
                                    <span>Learn More</span>
                                </button>
                            </div>
                        </div>

                        {/* Stats Grid */}
                        <div className="b-stats-grid">
                            {stats.map((stat, index) => {
                                const IconComponent = stat.icon;
                                return (
                                    <div key={index} className="b-stat-card" style={{ animationDelay: `${index * 0.1}s` }}>
                                        <div className="b-stat-icon">
                                            <IconComponent />
                                        </div>
                                        <div className="b-stat-number">{stat.number}</div>
                                        <div className="b-stat-label">{stat.label}</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </section>

                {/* How It Works */}
                <section className="b-process-section">
                    <div className="b-section-container">
                        <div className="b-section-header">
                            <h2 className="b-section-title">How It Works</h2>
                            <p className="b-section-subtitle">
                                Our AI-powered platform makes medical imaging analysis accessible to everyone
                            </p>
                        </div>

                        <div className="b-process-steps">
                            <div className="b-step-card b-upload-step">
                                <div className="b-step-icon">
                                    <Upload />
                                </div>
                                <h3 className="b-step-title">Upload Your Scan</h3>
                                <p className="b-step-description">
                                    Simply upload your X-ray or CT scan image. Our system automatically detects the image type and routes it to the appropriate AI model.
                                </p>
                                <div className="b-step-number">01</div>
                            </div>

                            <div className="b-step-connector"></div>

                            <div className="b-step-card b-analysis-step">
                                <div className="b-step-icon">
                                    <Brain />
                                </div>
                                <h3 className="b-step-title">AI Analysis</h3>
                                <p className="b-step-description">
                                    Our trained models analyze your image in seconds, detecting patterns and abnormalities with medical-grade accuracy.
                                </p>
                                <div className="b-step-number">02</div>
                            </div>

                            <div className="b-step-connector"></div>

                            <div className="b-step-card b-results-step">
                                <div className="b-step-icon">
                                    <FileText />
                                </div>
                                <h3 className="b-step-title">Get Results</h3>
                                <p className="b-step-description">
                                    Receive detailed results with confidence scores, expert summaries, and actionable insights - all in an easy-to-understand format.
                                </p>
                                <div className="b-step-number">03</div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* AI Models Showcase */}
                <section className="b-models-section">
                    <div className="b-section-container">
                        <div className="b-section-header">
                            <h2 className="b-section-title">Our AI Models</h2>
                            <p className="b-section-subtitle">
                                Specialized models trained on thousands of medical images to deliver precise diagnoses
                            </p>
                        </div>

                        <div className="b-models-showcase">
                            {/* Model Selection */}
                            <div className="b-models-list">
                                {Object.entries(models).map(([key, model], index) => (
                                    <div
                                        key={key}
                                        className={`b-model-card ${activeModel === key ? 'b-active' : ''}`}
                                        onClick={() => setActiveModel(key)}
                                        style={{ animationDelay: `${index * 0.1}s` }}
                                    >
                                        <div className="b-model-header">
                                            <h3 className="b-model-name">{model.name}</h3>
                                            <span className="b-model-accuracy">{model.accuracy}</span>
                                        </div>
                                        <p className="b-model-type">{model.type}</p>
                                        <p className="b-model-description">{model.description}</p>
                                        <div className="b-model-indicator"></div>
                                    </div>
                                ))}
                            </div>

                            {/* Model Details */}
                            <div className="b-model-details">
                                <div className="b-details-header">
                                    <h3 className="b-details-title">{models[activeModel].name}</h3>
                                    <div className="b-details-badges">
                                        <span className="b-badge b-type-badge">{models[activeModel].type}</span>
                                        <span className="b-badge b-accuracy-badge">{models[activeModel].accuracy} Accuracy</span>
                                    </div>
                                </div>
                                <p className="b-details-description">{models[activeModel].description}</p>
                                
                                <div className="b-features-list">
                                    <h4 className="b-features-title">Key Features:</h4>
                                    {models[activeModel].features.map((feature, index) => (
                                        <div key={index} className="b-feature-item">
                                            <Check className="b-feature-check" />
                                            <span>{feature}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Value Proposition */}
                <section className="b-value-section">
                    <div className="b-section-container">
                        <div className="b-value-content">
                            <div className="b-value-text">
                                <h2 className="b-value-title">Bridging the Healthcare Gap</h2>
                                <p className="b-value-description">
                                    DeepDiagnose democratizes access to medical imaging expertise, especially crucial for remote areas with limited healthcare infrastructure.
                                </p>
                                <div className="b-value-points">
                                    <div className="b-value-point">
                                        <Users className="b-point-icon" />
                                        <div className="b-point-content">
                                            <h3 className="b-point-title">For Everyone</h3>
                                            <p className="b-point-description">Accessible to both general public and medical professionals</p>
                                        </div>
                                    </div>
                                    <div className="b-value-point">
                                        <Zap className="b-point-icon" />
                                        <div className="b-point-content">
                                            <h3 className="b-point-title">Instant Results</h3>
                                            <p className="b-point-description">Get comprehensive analysis in seconds, not days</p>
                                        </div>
                                    </div>
                                    <div className="b-value-point">
                                        <Shield className="b-point-icon" />
                                        <div className="b-point-content">
                                            <h3 className="b-point-title">Completely Free</h3>
                                            <p className="b-point-description">No hidden costs, no subscriptions - healthcare should be accessible</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="b-value-cta">
                                <div className="b-cta-content">
                                    <div className="b-cta-icon">
                                        <Activity />
                                    </div>
                                    <h3 className="b-cta-title">Ready to Get Started?</h3>
                                    <p className="b-cta-description">
                                        Upload your medical scan and experience the power of AI-driven diagnostics
                                    </p>
                                    <button className="b-cta-button" onClick={handleClick}>
                                        Try It Now
                                        <ArrowRight />
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Developer API Section */}
                <section className="b-api-section">
                    <div className="b-section-container">
                        <div className="b-section-header">
                            <h2 className="b-section-title">Developer API</h2>
                            <p className="b-section-subtitle">
                                Integrate our AI models into your applications with our comprehensive API
                            </p>
                        </div>

                        <div className="b-api-content">
                            <div className="b-api-demo">
                                <div className="b-code-window">
                                    <div className="b-window-header">
                                        <div className="b-window-controls">
                                            <span className="b-control b-close"></span>
                                            <span className="b-control b-minimize"></span>
                                            <span className="b-control b-maximize"></span>
                                        </div>
                                        <div className="b-window-title">
                                            <Code />
                                            <span>API Example</span>
                                        </div>
                                    </div>
                                    <div className="b-code-content">
                                        <pre className="b-code-block">
{`POST /api/analyze
{
    "image": "base64_encoded_image",
}

Response:
{
    "scan_type: "X-ray",
    "confidence": 0.942,
    "anatomy": "Chest",
    "confidence": 0.942,
    "prediction": "pneumonia",
    "prediction_confidence": 0.942,
    "heatmap": "base64_encoded_heatmap",
    "summary": "Moderate pneumonia detected..."
}`}
                                        </pre>
                                    </div>
                                </div>
                                <div className="b-api-features">
                                    <div className="b-api-feature">
                                        <Check />
                                        <span>RESTful API with JSON responses</span>
                                    </div>
                                    <div className="b-api-feature">
                                        <Check />
                                        <span>Comprehensive documentation</span>
                                    </div>
                                    <div className="b-api-feature">
                                        <Check />
                                        <span>Rate limiting and authentication</span>
                                    </div>
                                </div>
                            </div>

                            <div className="b-api-info">
                                <h3 className="b-api-title">Build the Future of Healthcare</h3>
                                <p className="b-api-description">
                                    Leverage our AI models to create innovative healthcare solutions. Perfect for hospitals, clinics, and health tech startups.
                                </p>
                                <button className="b-api-button">
                                    View API Docs
                                    <ArrowRight />
                                </button>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
            
            {/* Auth Modals */}
            {showAuth && (
                <>
                    {authMode === 'signin' ? (
                        <SignIn 
                            onToggleAuth={handleToggleAuth}
                            onClose={handleCloseAuth}
                        />
                    ) : (
                        <SignUp 
                            onToggleAuth={handleToggleAuth}
                            onClose={handleCloseAuth}
                        />
                    )}
                </>
            )}
        </>
    );
};

export default Body;