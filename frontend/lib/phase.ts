// Constants and type definitions representing phase states
// Centralize the phase strings used by the UI and API responses

export const PHASE = {
  HEARING: 'hearing',
  SUMMARIZING: 'summarizing',
  CODE_GENERATION: 'code_generation',
  CODE_VALIDATION: 'code_validation',
  COMPLETED: 'completed',
} as const

export type Phase = typeof PHASE[keyof typeof PHASE]

export const isCompletedPhase = (p?: string | null): p is typeof PHASE.COMPLETED => p === PHASE.COMPLETED
