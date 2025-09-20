import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

// Import translation files
import jaTranslation from './locales/ja.json'
import enTranslation from './locales/en.json'

const resources = {
  ja: {
    translation: jaTranslation,
  },
  en: {
    translation: enTranslation,
  },
}

i18n
  .use(initReactI18next) // passes i18n down to react-i18next
  .init({
    resources,
    lng: 'en', // default language
    fallbackLng: 'ja',
    
    interpolation: {
      escapeValue: false, // react already does escaping
    },
  })

export default i18n
