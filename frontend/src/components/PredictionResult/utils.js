// utils.js - Helper functions for the application

/**
 * Formats a date object into a readable string format
 * Example output: "May 11, 2025 - 14:30"
 * 
 * @param {Date} date - The date to format
 * @returns {string} Formatted date string
 */
export const formatDate = (date) => {
  const options = { 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  };
  
  return date.toLocaleDateString('en-US', options);
};

/**
 * Truncates text to a specified length with ellipsis
 * 
 * @param {string} text - The text to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength) => {
  if (!text || text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
};

/**
 * Formats a confidence value to a percentage with 2 decimal places
 * 
 * @param {number} value - Confidence value between 0 and 1
 * @returns {string} Formatted percentage
 */
export const formatConfidence = (value) => {
  return (value * 100).toFixed(2) + '%';
};

/**
 * Calculates confidence level category based on value
 * 
 * @param {number} value - Confidence value between 0 and 1
 * @returns {string} Confidence level category
 */
export const getConfidenceLevel = (value) => {
  if (value >= 0.9) return 'very-high';
  if (value >= 0.75) return 'high';
  if (value >= 0.5) return 'medium';
  if (value >= 0.25) return 'low';
  return 'very-low';
};

/**
 * Generates a unique ID for components
 * 
 * @returns {string} Unique ID
 */
export const generateUniqueId = () => {
  return 'id-' + Math.random().toString(36).substring(2, 9);
};