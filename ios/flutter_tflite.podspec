#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_tflite.podspec' to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_tflite'
  s.version          = '0.0.1'
  s.summary          = 'Flutter Tensorflow Lite plugin.'
  s.description      = <<-DESC
A new flutter plugin project.
                       DESC
  s.homepage         = ''
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => '' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '9.0'

  # Flutter.framework does not contain a i386 slice. Only x86_64 simulators are supported.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'x86_64' }
  s.swift_version = '5.0'
end
