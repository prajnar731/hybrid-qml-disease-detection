/**
 * Centralized API Client Module — Stage 12.
 * Communicates exclusively with the FastAPI backend at http://127.0.0.1:8000.
 *
 * Provides standardized error handling, loading state helpers, and guarantees
 * that all experimental data, metrics, and explanations are sourced dynamically
 * from the backend without client-side hardcoding.
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);

    let data = null;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      const errorMsg = (data && data.detail)
        ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
        : `Request failed with status ${response.status} (${response.statusText})`;
      throw new ApiError(errorMsg, response.status, data);
    }

    return data;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    // Network errors (e.g. backend offline or refused connection)
    throw new ApiError(
      `Unable to connect to backend at ${API_BASE_URL}. Ensure FastAPI is running on port 8000.`,
      0,
      { originalError: err.message }
    );
  }
}

/**
 * Health check endpoint.
 * GET /health
 */
export async function getHealth() {
  return request('/health');
}

/**
 * Model specifications and catalog endpoint.
 * GET /models
 */
export async function getModels() {
  return request('/models');
}

/**
 * Consolidated Stage 10 benchmark results.
 * GET /benchmark
 */
export async function getBenchmark() {
  return request('/benchmark');
}

/**
 * Adaptive quantum resource optimization results.
 * GET /selection
 */
export async function getSelection() {
  return request('/selection');
}

/**
 * Quantum gate-level noise experiment results.
 * GET /noise
 */
export async function getNoise() {
  return request('/noise');
}

/**
 * Single-sample disease risk screening.
 * POST /predict
 * @param {Object} payload { model: string, features: Object }
 */
export async function predict(payload) {
  return request('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
