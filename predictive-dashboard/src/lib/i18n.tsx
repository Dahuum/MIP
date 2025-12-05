'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type Language = 'en' | 'fr';

interface Translations {
  // Header
  header: {
    dashboard: string;
    predictiveMaintenance: string;
    live: string;
    analysis: string;
    model: string;
    language: string;
    theme: string;
    dark: string;
    light: string;
  };
  // Risk levels
  risk: {
    low: string;
    medium: string;
    high: string;
    critical: string;
    riskLevel: string;
    failureProbability: string;
    riskProbability: string;
  };
  // Sensors
  sensors: {
    title: string;
    motorCurrent: string;
    tempOpposite: string;
    tempMotor: string;
    vibOpposite: string;
    vibMotor: string;
    valveOpening: string;
    solidRate: string;
    pumpFlowRate: string;
    reset: string;
    adjustSensors: string;
    // Sensor name map by key
    sensorNames: Record<string, string>;
  };
  // Scenarios
  scenarios: {
    title: string;
    selectScenario: string;
    demonstrationScenarios: string;
    selectToSimulate: string;
    active: string;
    normal: string;
    normalDesc: string;
    warning: string;
    warningDesc: string;
    critical: string;
    criticalDesc: string;
    motorOverheat: string;
    motorOverheatDesc: string;
    pumpCavitation: string;
    pumpCavitationDesc: string;
    valveBlocked: string;
    valveBlockedDesc: string;
    highVibration: string;
    highVibrationDesc: string;
    dustStorm: string;
    dustStormDesc: string;
    bearingWear: string;
    bearingWearDesc: string;
    // Scenario names map by ID
    scenarioNames: Record<string, string>;
  };
  // Alerts
  alerts: {
    title: string;
    noAlerts: string;
    systemNormal: string;
    attention: string;
    warning: string;
    critical: string;
    imbalance: string;
    dustProbability: string;
    timeToFailure: string;
    savings: string;
    topContributingFactors: string;
    recommendedActions: string;
    detected: string;
  };
  // Stats
  stats: {
    totalInteractions: string;
    highRiskEvents: string;
    criticalAlerts: string;
    maxRiskReached: string;
    sessionDuration: string;
    testsPerformed: string;
    riskHistory: string;
    featureImportance: string;
    sensorHistory: string;
    failuresPrevented: string;
    totalSavings: string;
    modelRecall: string;
    systemUptime: string;
    daysSinceFailure: string;
    vsLastMonth: string;
    vsProjection: string;
    returnOnInvestment: string;
    failuresCaught: string;
    roi: string;
    paybackPeriod: string;
    months: string;
  };
  // Diagnostic messages
  diagnosticMessages: {
    powerLoss: string;
    electricalCircuit: string;
    fanOverheating: string;
    bearingIssue: string;
    bearingAxleProblem: string;
    imbalanceDetected: string;
    imbalanceDust: string;
    imbalanceElevatedSolid: string;
    multipleIssues: string;
    systemNormal: string;
    warningSignsDetected: string;
    elevatedRiskDetected: string;
    criticalConditions: string;
  };
  // Model info
  modelInfo: {
    title: string;
    architecture: string;
    accuracy: string;
    lastTrained: string;
    dataPoints: string;
    features: string;
    lstmLayers: string;
    hiddenUnits: string;
    trainedOn: string;
    realFailures: string;
    precision: string;
    recall: string;
    f1Score: string;
    whatAiLearned: string;
    vibration: string;
    temperature: string;
    current: string;
    category: string;
    importance: string;
    howAiPredicts: string;
    // Architecture details
    bidirectionalLstm: string;
    inputFeatures: string;
    parameters: string;
    predictionWindow: string;
    input: string;
    units: string;
    sensorData: string;
    temporalPatterns: string;
    featureLearning: string;
    classification: string;
    output: string;
    riskScore: string;
    imbalanceDetection: string;
    // How AI works
    dataCollection: string;
    dataCollectionDesc: string;
    patternRecognition: string;
    patternRecognitionDesc: string;
    earlyWarning: string;
    earlyWarningDesc: string;
    // Feature importance
    changes: string;
    avgDelta: string;
    notTestedYet: string;
    startTesting: string;
    goToLive: string;
    // Sensor history
    risk: string;
    // Historic failures table
    source: string;
    date: string;
    time: string;
    type: string;
    duration: string;
    description: string;
    historicalRecords: string;
    sessionCriticalErrors: string;
    session: string;
    historical: string;
    vibrationIssues: string;
    temperatureIssues: string;
    // Buttons
    resetAllData: string;
    viewTrainingStats: string;
    noDataYet: string;
    // Failure types mapping
    failureTypes: Record<string, string>;
    failureDescriptions: Record<string, string>;
  };
  // Dashboard sections
  dashboard: {
    liveSimulation: string;
    riskAnalysis: string;
    systemStatus: string;
    sensorStatus: string;
    quickScenarios: string;
    historicalAnalysis: string;
    riskTrends: string;
    sessionInsights: string;
    dynamicFeatureImportance: string;
    whatMattersNow: string;
    sensorImpactLog: string;
    recentChanges: string;
    historicFailures: string;
    pastIncidents: string;
    modelPerformance: string;
    keyMetrics: string;
    trainingData: string;
    learnedPatterns: string;
    line307FanC07: string;
    preventingLosses: string;
    featureImportance: string;
    sensorHistory: string;
  };
  // General
  general: {
    loading: string;
    error: string;
    save: string;
    cancel: string;
    apply: string;
    close: string;
    search: string;
    noData: string;
    hours: string;
    minutes: string;
    seconds: string;
    ago: string;
    from: string;
    to: string;
    change: string;
    impact: string;
  };
}

const translations: Record<Language, Translations> = {
  en: {
    header: {
      dashboard: 'Dashboard',
      predictiveMaintenance: 'Predictive Maintenance',
      live: 'Live',
      analysis: 'Analysis',
      model: 'Model',
      language: 'Language',
      theme: 'Theme',
      dark: 'Dark',
      light: 'Light',
    },
    risk: {
      low: 'Low Risk',
      medium: 'Moderate Risk',
      high: 'High Risk',
      critical: 'Critical',
      riskLevel: 'Risk Level',
      failureProbability: 'Failure Probability',
      riskProbability: 'Risk Probability',
    },
    sensors: {
      title: 'Sensors',
      motorCurrent: 'Motor Current',
      tempOpposite: 'Temp (Opposite)',
      tempMotor: 'Temp (Motor)',
      vibOpposite: 'Vib (Opposite)',
      vibMotor: 'Vib (Motor)',
      valveOpening: 'Valve Opening',
      solidRate: 'Solid Rate',
      pumpFlowRate: 'Pump Flow Rate',
      reset: 'Reset',
      adjustSensors: 'Adjust Sensors',
      sensorNames: {
        'motor_current': 'Motor Current',
        'temp_opposite': 'Temp (Opposite)',
        'temp_motor': 'Temp (Motor)',
        'vib_opposite': 'Vib (Opposite)',
        'vib_motor': 'Vib (Motor)',
        'valve_opening': 'Valve Opening',
        'solid_rate': 'Solid Rate',
        'pump_flow_rate': 'Pump Flow Rate',
        'dust_concentration': 'Dust Concentration',
        'vibration_level': 'Vibration Level',
        'temperature': 'Temperature',
        'current': 'Current',
        'flow_rate': 'Flow Rate',
      },
    },
    scenarios: {
      title: 'Scenarios',
      selectScenario: 'Select Scenario',
      demonstrationScenarios: 'Demonstration Scenarios',
      selectToSimulate: 'Select a scenario to simulate different failure conditions',
      active: 'Active',
      normal: 'Normal Operation',
      normalDesc: 'All systems functioning within optimal parameters',
      warning: 'Warning State',
      warningDesc: 'Some parameters approaching threshold limits',
      critical: 'Critical State',
      criticalDesc: 'Multiple parameters exceeding safe limits',
      motorOverheat: 'Motor Overheat',
      motorOverheatDesc: 'Motor temperature rising above safe levels',
      pumpCavitation: 'Pump Cavitation',
      pumpCavitationDesc: 'Flow rate issues indicating cavitation',
      valveBlocked: 'Valve Blocked',
      valveBlockedDesc: 'Valve opening restricted',
      highVibration: 'High Vibration',
      highVibrationDesc: 'Excessive vibration detected',
      dustStorm: 'Dust Storm',
      dustStormDesc: 'High dust concentration affecting equipment',
      bearingWear: 'Bearing Wear',
      bearingWearDesc: 'High vibration pattern typical of bearing degradation',
      scenarioNames: {
        'normal': 'Normal Operation',
        'imbalance': 'Imbalance Accumulation',
        'warning_signs': 'Warning Signs',
        'high_risk': 'High Risk',
        'critical_failure': 'Critical Failure',
        'high_temp': 'High Temperature',
        'bearing_wear': 'Bearing Wear',
      },
    },
    alerts: {
      title: 'Alerts',
      noAlerts: 'No active alerts',
      systemNormal: 'System operating normally',
      attention: 'Attention Required',
      warning: 'Warning',
      critical: 'Critical Alert',
      imbalance: 'Imbalance',
      dustProbability: 'dust probability',
      timeToFailure: 'Time to Failure',
      savings: 'Savings',
      topContributingFactors: 'Top Contributing Factors',
      recommendedActions: 'Recommended Actions',
      detected: 'Detected',
    },
    stats: {
      totalInteractions: 'Total Interactions',
      highRiskEvents: 'High Risk Events',
      criticalAlerts: 'Critical Alerts',
      maxRiskReached: 'Max Risk Reached',
      sessionDuration: 'Session Duration',
      testsPerformed: 'Tests Performed',
      riskHistory: 'Risk History',
      featureImportance: 'Feature Importance',
      sensorHistory: 'Sensor History',
      failuresPrevented: 'Failures Prevented',
      totalSavings: 'Total Savings',
      modelRecall: 'Model Recall',
      systemUptime: 'System Uptime',
      daysSinceFailure: 'Days Since Failure',
      vsLastMonth: 'vs last month',
      vsProjection: 'vs projection',
      returnOnInvestment: 'Return on Investment',
      failuresCaught: 'Failures Caught',
      roi: 'ROI',
      paybackPeriod: 'Payback Period',
      months: 'months',
    },
    diagnosticMessages: {
      powerLoss: '⚡ POWER LOSS: Check power distribution system',
      electricalCircuit: '🔌 ELECTRICAL CIRCUIT PROBLEM: Check motor electrical connections',
      fanOverheating: '🔥 FAN OVERHEATING: Check cooling system and ventilation',
      bearingIssue: '⚙️ BEARING ISSUE: Lack of grease or starting imbalance',
      bearingAxleProblem: '🔧 BEARING/AXLE PROBLEM: Check bearing alignment and axle condition',
      imbalanceDetected: '🌫️ IMBALANCE DETECTED: Dust accumulation due to low pump flow',
      imbalanceDust: '🌫️ IMBALANCE DETECTED: Dust in system',
      imbalanceElevatedSolid: '⚠️ IMBALANCE DETECTING: Elevated solid rate',
      multipleIssues: '⚠️ MULTIPLE ISSUES: Mechanical problem combined with dust accumulation',
      systemNormal: '✅ System operating normally',
      warningSignsDetected: '⚠️ Warning signs detected - Increase monitoring frequency',
      elevatedRiskDetected: '🔶 Elevated risk detected - Schedule maintenance inspection',
      criticalConditions: '🚨 Critical conditions - Immediate attention required',
    },
    modelInfo: {
      title: 'Model Information',
      architecture: 'Architecture',
      accuracy: 'Accuracy',
      lastTrained: 'Last Trained',
      dataPoints: 'Data Points',
      features: 'Features',
      lstmLayers: 'LSTM Layers',
      hiddenUnits: 'Hidden Units',
      trainedOn: 'Trained On',
      realFailures: 'Real Failures',
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1 Score',
      whatAiLearned: 'What the AI Learned',
      vibration: 'Vibration',
      temperature: 'Temperature',
      current: 'Current',
      category: 'Category',
      importance: 'Importance',
      howAiPredicts: 'How the AI Predicts Failures',
      // Architecture details
      bidirectionalLstm: 'Bidirectional LSTM',
      inputFeatures: 'input features',
      parameters: 'Parameters',
      predictionWindow: 'Prediction Window',
      input: 'Input',
      units: 'units',
      sensorData: 'Sensor Data',
      temporalPatterns: 'Temporal Patterns',
      featureLearning: 'Feature Learning',
      classification: 'Classification',
      output: 'Output',
      riskScore: 'Risk Score',
      imbalanceDetection: 'Imbalance Detection',
      // How AI works
      dataCollection: 'Data Collection',
      dataCollectionDesc: 'Sensors continuously monitor vibration, temperature, and current from Fan C07. Data is collected every minute.',
      patternRecognition: 'Pattern Recognition',
      patternRecognitionDesc: 'The LSTM learns temporal patterns from 22 historical failures. It identifies the signature of impending failures.',
      earlyWarning: 'Early Warning',
      earlyWarningDesc: 'When patterns match pre-failure conditions, alerts are triggered 24 hours before failure occurs.',
      // Feature importance
      changes: 'changes',
      avgDelta: 'avg',
      notTestedYet: 'Not tested yet',
      startTesting: 'Start testing sensors to see which ones cause the most risk!',
      goToLive: 'Go to Live Simulation tab and adjust the sliders',
      // Sensor history
      risk: 'risk',
      // Historic failures table
      source: 'Source',
      date: 'Date',
      time: 'Time',
      type: 'Type',
      duration: 'Duration',
      description: 'Description',
      historicalRecords: 'Historical Records',
      sessionCriticalErrors: 'Session Critical Errors',
      session: 'Session',
      historical: 'Historical',
      vibrationIssues: 'Vibration Issues',
      temperatureIssues: 'Temperature Issues',
      // Buttons
      resetAllData: 'Reset All Data',
      viewTrainingStats: 'View Training Stats',
      noDataYet: 'No failure data available',
      // Failure types mapping (for translating API responses)
      failureTypes: {
        'VIBRATION': 'Vibration',
        'TEMPERATURE': 'Temperature',
        'BEARING': 'Bearing',
        'MECHANICAL': 'Mechanical',
        'STRUCTURAL': 'Structural',
        'TURBINE': 'Turbine',
        'VALVE': 'Valve',
        'DUST_ACCUMULATION': 'Dust Accumulation',
        'OVERHEATING': 'Overheating',
        'BEARING_AXLE': 'Bearing/Axle',
        'CRITICAL': 'Critical',
        'MOTOR': 'Motor',
        'ELECTRICAL': 'Electrical',
        'HYDRAULIC': 'Hydraulic',
        'PNEUMATIC': 'Pneumatic',
      } as Record<string, string>,
      // Failure descriptions mapping
      failureDescriptions: {
        'High vibration detected on fan motor': 'High vibration detected on fan motor',
        'Abnormal temperature spike detected': 'Abnormal temperature spike detected',
        'Bearing wear detected in motor assembly': 'Bearing wear detected in motor assembly',
        'Mechanical failure in drive system': 'Mechanical failure in drive system',
        'Structural stress detected': 'Structural stress detected',
        'Turbine blade imbalance': 'Turbine blade imbalance',
        'Valve malfunction detected': 'Valve malfunction detected',
        'Dust accumulation in air intake': 'Dust accumulation in air intake',
        'Critical overheating in main unit': 'Critical overheating in main unit',
        'Bearing axle misalignment': 'Bearing axle misalignment',
        // Diagnostic messages from API
        '⚡ POWER LOSS: Check power distribution system': '⚡ POWER LOSS: Check power distribution system',
        '🔌 ELECTRICAL CIRCUIT PROBLEM: Check motor electrical connections': '🔌 ELECTRICAL CIRCUIT PROBLEM: Check motor electrical connections',
        '🔥 FAN OVERHEATING: Check cooling system and ventilation': '🔥 FAN OVERHEATING: Check cooling system and ventilation',
        '⚙️ BEARING ISSUE: Lack of grease or starting imbalance': '⚙️ BEARING ISSUE: Lack of grease or starting imbalance',
        '🔧 BEARING/AXLE PROBLEM: Check bearing alignment and axle condition': '🔧 BEARING/AXLE PROBLEM: Check bearing alignment and axle condition',
        '⚠️ MULTIPLE ISSUES: Mechanical problem combined with dust accumulation': '⚠️ MULTIPLE ISSUES: Mechanical problem combined with dust accumulation',
        '✅ System operating normally': '✅ System operating normally',
        '⚠️ Warning signs detected - Increase monitoring frequency': '⚠️ Warning signs detected - Increase monitoring frequency',
        '🔶 Elevated risk detected - Schedule maintenance inspection': '🔶 Elevated risk detected - Schedule maintenance inspection',
        '🚨 Critical conditions - Immediate attention required': '🚨 Critical conditions - Immediate attention required',
      } as Record<string, string>,
    },
    dashboard: {
      liveSimulation: 'Live Simulation',
      riskAnalysis: 'Risk Analysis',
      systemStatus: 'System Status',
      sensorStatus: 'Sensor Status',
      quickScenarios: 'Quick Scenarios',
      historicalAnalysis: 'Historical Analysis',
      riskTrends: 'Risk Trends',
      sessionInsights: 'Session Insights',
      dynamicFeatureImportance: 'Dynamic Feature Importance',
      whatMattersNow: 'What Matters Now',
      sensorImpactLog: 'Sensor Impact Log',
      recentChanges: 'Recent Changes',
      historicFailures: 'Historic Failures',
      pastIncidents: 'Past Incidents',
      modelPerformance: 'Model Performance',
      keyMetrics: 'Key Metrics',
      trainingData: 'Training Data',
      learnedPatterns: 'Learned Patterns',
      line307FanC07: 'Fan C07',
      preventingLosses: 'Preventing $4.4M+ in annual losses',
      featureImportance: 'Feature Importance',
      sensorHistory: 'Sensor History',
    },
    general: {
      loading: 'Loading...',
      error: 'Error',
      save: 'Save',
      cancel: 'Cancel',
      apply: 'Apply',
      close: 'Close',
      search: 'Search',
      noData: 'No data available',
      hours: 'hours',
      minutes: 'minutes',
      seconds: 'seconds',
      ago: 'ago',
      from: 'from',
      to: 'to',
      change: 'Change',
      impact: 'Impact',
    },
  },
  fr: {
    header: {
      dashboard: 'Tableau de bord',
      predictiveMaintenance: 'Maintenance Prédictive',
      live: 'Direct',
      analysis: 'Analyse',
      model: 'Modèle',
      language: 'Langue',
      theme: 'Thème',
      dark: 'Sombre',
      light: 'Clair',
    },
    risk: {
      low: 'Risque Faible',
      medium: 'Risque Modéré',
      high: 'Risque Élevé',
      critical: 'Critique',
      riskLevel: 'Niveau de Risque',
      failureProbability: 'Probabilité de Panne',
      riskProbability: 'Probabilité de Risque',
    },
    sensors: {
      title: 'Capteurs',
      motorCurrent: 'Courant Moteur',
      tempOpposite: 'Temp (Opposée)',
      tempMotor: 'Temp (Moteur)',
      vibOpposite: 'Vib (Opposée)',
      vibMotor: 'Vib (Moteur)',
      valveOpening: 'Ouverture Vanne',
      solidRate: 'Taux de Solide',
      pumpFlowRate: 'Débit Pompe',
      reset: 'Réinitialiser',
      adjustSensors: 'Ajuster les Capteurs',
      sensorNames: {
        'motor_current': 'Courant Moteur',
        'temp_opposite': 'Temp (Opposée)',
        'temp_motor': 'Temp (Moteur)',
        'vib_opposite': 'Vib (Opposée)',
        'vib_motor': 'Vib (Moteur)',
        'valve_opening': 'Ouverture Vanne',
        'solid_rate': 'Taux de Solide',
        'pump_flow_rate': 'Débit Pompe',
        'dust_concentration': 'Concentration de Poussière',
        'vibration_level': 'Niveau de Vibration',
        'temperature': 'Température',
        'current': 'Courant',
        'flow_rate': 'Débit',
      },
    },
    scenarios: {
      title: 'Scénarios',
      selectScenario: 'Sélectionner un Scénario',
      demonstrationScenarios: 'Scénarios de Démonstration',
      selectToSimulate: 'Sélectionnez un scénario pour simuler différentes conditions de panne',
      active: 'Actif',
      normal: 'Fonctionnement Normal',
      normalDesc: 'Tous les systèmes fonctionnent dans les paramètres optimaux',
      warning: 'État d\'Avertissement',
      warningDesc: 'Certains paramètres approchent des limites de seuil',
      critical: 'État Critique',
      criticalDesc: 'Plusieurs paramètres dépassent les limites de sécurité',
      motorOverheat: 'Surchauffe Moteur',
      motorOverheatDesc: 'Température du moteur au-dessus des niveaux de sécurité',
      pumpCavitation: 'Cavitation Pompe',
      pumpCavitationDesc: 'Problèmes de débit indiquant une cavitation',
      valveBlocked: 'Vanne Bloquée',
      valveBlockedDesc: 'Ouverture de vanne restreinte',
      highVibration: 'Vibration Élevée',
      highVibrationDesc: 'Vibration excessive détectée',
      dustStorm: 'Tempête de Poussière',
      dustStormDesc: 'Concentration élevée de poussière affectant l\'équipement',
      bearingWear: 'Usure des Roulements',
      bearingWearDesc: 'Vibration élevée typique de la dégradation des roulements',
      scenarioNames: {
        'normal': 'Fonctionnement Normal',
        'imbalance': 'Accumulation de Balourdclea',
        'warning_signs': 'Signes d\'Alerte',
        'high_risk': 'Risque Élevé',
        'critical_failure': 'Panne Critique',
        'high_temp': 'Haute Température',
        'bearing_wear': 'Usure des Roulements',
      },
    },
    alerts: {
      title: 'Alertes',
      noAlerts: 'Aucune alerte active',
      systemNormal: 'Système fonctionnant normalement',
      attention: 'Attention Requise',
      warning: 'Avertissement',
      critical: 'Alerte Critique',
      imbalance: 'Balourd',
      dustProbability: 'probabilité de poussière',
      timeToFailure: 'Temps avant Panne',
      savings: 'Économies',
      topContributingFactors: 'Principaux Facteurs Contributifs',
      recommendedActions: 'Actions Recommandées',
      detected: 'Détecté',
    },
    stats: {
      totalInteractions: 'Interactions Totales',
      highRiskEvents: 'Événements à Haut Risque',
      criticalAlerts: 'Alertes Critiques',
      maxRiskReached: 'Risque Maximum Atteint',
      sessionDuration: 'Durée de Session',
      testsPerformed: 'Tests Effectués',
      riskHistory: 'Historique des Risques',
      featureImportance: 'Importance des Caractéristiques',
      sensorHistory: 'Historique des Capteurs',
      failuresPrevented: 'Pannes Évitées',
      totalSavings: 'Économies Totales',
      modelRecall: 'Rappel du Modèle',
      systemUptime: 'Disponibilité Système',
      daysSinceFailure: 'Jours Sans Panne',
      vsLastMonth: 'vs mois dernier',
      vsProjection: 'vs projection',
      returnOnInvestment: 'Retour sur Investissement',
      failuresCaught: 'Pannes Détectées',
      roi: 'RSI',
      paybackPeriod: 'Période de Récupération',
      months: 'mois',
    },
    diagnosticMessages: {
      powerLoss: '⚡ PERTE DE COURANT : Vérifier le système de distribution électrique',
      electricalCircuit: '🔌 PROBLÈME DE CIRCUIT ÉLECTRIQUE : Vérifier les connexions électriques du moteur',
      fanOverheating: '🔥 SURCHAUFFE DU VENTILATEUR : Vérifier le système de refroidissement et la ventilation',
      bearingIssue: '⚙️ PROBLÈME DE ROULEMENT : Manque de graisse ou déséquilibre au démarrage',
      bearingAxleProblem: '🔧 PROBLÈME DE ROULEMENT/ESSIEU : Vérifier l\'alignement du roulement et l\'état de l\'essieu',
      imbalanceDetected: '🌫️ DÉSÉQUILIBRE DÉTECTÉ : Accumulation de poussière due à un faible débit de pompe',
      imbalanceDust: '🌫️ DÉSÉQUILIBRE DÉTECTÉ : Poussière dans le système',
      imbalanceElevatedSolid: '⚠️ DÉTECTION DE DÉSÉQUILIBRE : Taux de solide élevé',
      multipleIssues: '⚠️ PROBLÈMES MULTIPLES : Problème mécanique combiné à une accumulation de poussière',
      systemNormal: '✅ Système fonctionnant normalement',
      warningSignsDetected: '⚠️ Signes d\'alerte détectés - Augmenter la fréquence de surveillance',
      elevatedRiskDetected: '🔶 Risque élevé détecté - Planifier une inspection de maintenance',
      criticalConditions: '🚨 Conditions critiques - Attention immédiate requise',
    },
    modelInfo: {
      title: 'Informations du Modèle',
      architecture: 'Architecture',
      accuracy: 'Précision',
      lastTrained: 'Dernier Entraînement',
      dataPoints: 'Points de Données',
      features: 'Caractéristiques',
      lstmLayers: 'Couches LSTM',
      hiddenUnits: 'Unités Cachées',
      trainedOn: 'Entraîné Sur',
      realFailures: 'Pannes Réelles',
      precision: 'Précision',
      recall: 'Rappel',
      f1Score: 'Score F1',
      whatAiLearned: 'Ce que l\'IA a Appris',
      vibration: 'Vibration',
      temperature: 'Température',
      current: 'Courant',
      category: 'Catégorie',
      importance: 'Importance',
      howAiPredicts: 'Comment l\'IA Prédit les Pannes',
      // Architecture details
      bidirectionalLstm: 'LSTM Bidirectionnel',
      inputFeatures: 'caractéristiques d\'entrée',
      parameters: 'Paramètres',
      predictionWindow: 'Fenêtre de Prédiction',
      input: 'Entrée',
      units: 'unités',
      sensorData: 'Données Capteurs',
      temporalPatterns: 'Motifs Temporels',
      featureLearning: 'Apprentissage',
      classification: 'Classification',
      output: 'Sortie',
      riskScore: 'Score de Risque',
      imbalanceDetection: 'Détection de Balourd',
      // How AI works
      dataCollection: 'Collecte de Données',
      dataCollectionDesc: 'Les capteurs surveillent en continu les vibrations, la température et le courant du Ventilateur C07. Les données sont collectées chaque minute.',
      patternRecognition: 'Reconnaissance de Motifs',
      patternRecognitionDesc: 'Le LSTM apprend les motifs temporels à partir de 22 pannes historiques. Il identifie la signature des pannes imminentes.',
      earlyWarning: 'Alerte Précoce',
      earlyWarningDesc: 'Lorsque les motifs correspondent aux conditions de pré-panne, des alertes sont déclenchées 24 heures avant la panne.',
      // Feature importance
      changes: 'changements',
      avgDelta: 'moy',
      notTestedYet: 'Pas encore testé',
      startTesting: 'Commencez à tester les capteurs pour voir lesquels causent le plus de risque!',
      goToLive: 'Allez dans l\'onglet Simulation en Direct et ajustez les curseurs',
      // Sensor history
      risk: 'risque',
      // Historic failures table
      source: 'Source',
      date: 'Date',
      time: 'Heure',
      type: 'Type',
      duration: 'Durée',
      description: 'Description',
      historicalRecords: 'Enregistrements Historiques',
      sessionCriticalErrors: 'Erreurs Critiques de Session',
      session: 'Session',
      historical: 'Historique',
      vibrationIssues: 'Problèmes de Vibration',
      temperatureIssues: 'Problèmes de Température',
      // Buttons
      resetAllData: 'Réinitialiser les Données',
      viewTrainingStats: 'Voir Stats d\'Entraînement',
      noDataYet: 'Aucune donnée de panne disponible',
      // Failure types mapping (for translating API responses)
      failureTypes: {
        'VIBRATION': 'Vibration',
        'TEMPERATURE': 'Température',
        'BEARING': 'Roulement',
        'MECHANICAL': 'Mécanique',
        'STRUCTURAL': 'Structurel',
        'TURBINE': 'Turbine',
        'VALVE': 'Vanne',
        'DUST_ACCUMULATION': 'Accumulation de Poussière',
        'OVERHEATING': 'Surchauffe',
        'BEARING_AXLE': 'Roulement/Essieu',
        'CRITICAL': 'Critique',
        'MOTOR': 'Moteur',
        'ELECTRICAL': 'Électrique',
        'HYDRAULIC': 'Hydraulique',
        'PNEUMATIC': 'Pneumatique',
      } as Record<string, string>,
      // Failure descriptions mapping
      failureDescriptions: {
        'High vibration detected on fan motor': 'Vibration élevée détectée sur le moteur du ventilateur',
        'Abnormal temperature spike detected': 'Pic de température anormal détecté',
        'Bearing wear detected in motor assembly': 'Usure de roulement détectée dans l\'assemblage du moteur',
        'Mechanical failure in drive system': 'Défaillance mécanique dans le système d\'entraînement',
        'Structural stress detected': 'Contrainte structurelle détectée',
        'Turbine blade imbalance': 'Déséquilibre des pales de turbine',
        'Valve malfunction detected': 'Dysfonctionnement de vanne détecté',
        'Dust accumulation in air intake': 'Accumulation de poussière dans l\'admission d\'air',
        'Critical overheating in main unit': 'Surchauffe critique dans l\'unité principale',
        'Bearing axle misalignment': 'Désalignement de l\'essieu de roulement',
        // Diagnostic messages from API
        '⚡ POWER LOSS: Check power distribution system': '⚡ PERTE DE COURANT : Vérifier le système de distribution électrique',
        '🔌 ELECTRICAL CIRCUIT PROBLEM: Check motor electrical connections': '🔌 PROBLÈME DE CIRCUIT ÉLECTRIQUE : Vérifier les connexions électriques du moteur',
        '🔥 FAN OVERHEATING: Check cooling system and ventilation': '🔥 SURCHAUFFE DU VENTILATEUR : Vérifier le système de refroidissement et la ventilation',
        '⚙️ BEARING ISSUE: Lack of grease or starting imbalance': '⚙️ PROBLÈME DE ROULEMENT : Manque de graisse ou déséquilibre au démarrage',
        '🔧 BEARING/AXLE PROBLEM: Check bearing alignment and axle condition': '🔧 PROBLÈME DE ROULEMENT/ESSIEU : Vérifier l\'alignement du roulement et l\'état de l\'essieu',
        '⚠️ MULTIPLE ISSUES: Mechanical problem combined with dust accumulation': '⚠️ PROBLÈMES MULTIPLES : Problème mécanique combiné à une accumulation de poussière',
        '✅ System operating normally': '✅ Système fonctionnant normalement',
        '⚠️ Warning signs detected - Increase monitoring frequency': '⚠️ Signes d\'alerte détectés - Augmenter la fréquence de surveillance',
        '🔶 Elevated risk detected - Schedule maintenance inspection': '🔶 Risque élevé détecté - Planifier une inspection de maintenance',
        '🚨 Critical conditions - Immediate attention required': '🚨 Conditions critiques - Attention immédiate requise',
      } as Record<string, string>,
    },
    dashboard: {
      liveSimulation: 'Simulation en Direct',
      riskAnalysis: 'Analyse des Risques',
      systemStatus: 'État du Système',
      sensorStatus: 'État des Capteurs',
      quickScenarios: 'Scénarios Rapides',
      historicalAnalysis: 'Analyse Historique',
      riskTrends: 'Tendances des Risques',
      sessionInsights: 'Aperçus de Session',
      dynamicFeatureImportance: 'Importance Dynamique des Caractéristiques',
      whatMattersNow: 'Ce qui Compte Maintenant',
      sensorImpactLog: 'Journal d\'Impact des Capteurs',
      recentChanges: 'Changements Récents',
      historicFailures: 'Pannes Historiques',
      pastIncidents: 'Incidents Passés',
      modelPerformance: 'Performance du Modèle',
      keyMetrics: 'Métriques Clés',
      trainingData: 'Données d\'Entraînement',
      learnedPatterns: 'Modèles Appris',
      line307FanC07: 'Ventilateur C07',
      preventingLosses: 'Prévention de 4,4M$+ de pertes annuelles',
      featureImportance: 'Importance des Caractéristiques',
      sensorHistory: 'Historique des Capteurs',
    },
    general: {
      loading: 'Chargement...',
      error: 'Erreur',
      save: 'Sauvegarder',
      cancel: 'Annuler',
      apply: 'Appliquer',
      close: 'Fermer',
      search: 'Rechercher',
      noData: 'Aucune donnée disponible',
      hours: 'heures',
      minutes: 'minutes',
      seconds: 'secondes',
      ago: 'il y a',
      from: 'de',
      to: 'à',
      change: 'Changement',
      impact: 'Impact',
    },
  },
};

interface I18nContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: Translations;
}

const I18nContext = createContext<I18nContextType | undefined>(undefined);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>('en');

  const value = {
    language,
    setLanguage,
    t: translations[language],
  };

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (context === undefined) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}

export { translations };
export type { Translations };
