'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Car,
  Fuel,
  Calendar,
  DollarSign,
  Heart,
  GitCompare,
  CreditCard,
  Star,
  MapPin,
  Settings,
  Shield
} from 'lucide-react';

interface Vehicle {
  id: string;
  brand: string;
  model: string;
  year: number;
  price: number;
  mileage: number;
  fuel_type: string;
  body_type: string;
  color: string;
  location: string;
  images: string[];
  features: string[];
  fuel_efficiency: number;
  safety_rating: number;
  match_score: number;
  description?: string;
}

interface UserProfile {
  user_id: string;
  name: string;
  email: string;
  age: number;
  income: number;
  preferences: string[];
  purpose: string;
}

interface VehicleGridProps {
  vehicles: Vehicle[];
  onVehicleSelect: (vehicleId: string) => void;
  onRequestFinancing: (vehicleId: string) => void;
  selectedVehicles: string[];
  onVehicleInteraction: (vehicleId: string, interactionType: 'click' | 'save' | 'view') => void;
  userProfile: UserProfile;
}

export function VehicleGrid({
  vehicles,
  onVehicleSelect,
  onRequestFinancing,
  selectedVehicles,
  onVehicleInteraction,
  userProfile
}: VehicleGridProps) {
  const [favoriteVehicles, setFavoriteVehicles] = useState<string[]>([]);

  const toggleFavorite = (vehicleId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavoriteVehicles(prev =>
      prev.includes(vehicleId)
        ? prev.filter(id => id !== vehicleId)
        : [...prev, vehicleId]
    );
    onVehicleInteraction(vehicleId, 'save');
  };

  const handleVehicleClick = (vehicleId: string) => {
    onVehicleInteraction(vehicleId, 'view');
    onVehicleSelect(vehicleId);
  };

  const getFuelTypeInfo = (fuelType: string) => {
    const fuelMap: Record<string, { label: string; color: string; emoji: string }> = {
      gasoline: { label: '가솔린', color: 'bg-blue-100 text-blue-800', emoji: '⛽' },
      diesel: { label: '디젤', color: 'bg-green-100 text-green-800', emoji: '🛢️' },
      hybrid: { label: '하이브리드', color: 'bg-emerald-100 text-emerald-800', emoji: '🔋' },
      electric: { label: '전기', color: 'bg-purple-100 text-purple-800', emoji: '⚡' },
    };
    return fuelMap[fuelType] || { label: fuelType, color: 'bg-gray-100 text-gray-800', emoji: '🚗' };
  };

  const getMatchScoreColor = (score: number) => {
    if (score >= 90) return 'from-emerald-500 to-green-500';
    if (score >= 80) return 'from-purple-500 to-pink-500';
    if (score >= 70) return 'from-yellow-500 to-orange-500';
    return 'from-gray-500 to-slate-500';
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
          🎯 AI 맞춤 차량 추천
        </h2>
        <p className="text-gray-600 text-lg">
          총 {vehicles.length}대의 차량이 당신의 조건과 매칭되었습니다
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {vehicles.map((vehicle) => {
          const fuelInfo = getFuelTypeInfo(vehicle.fuel_type);
          const isSelected = selectedVehicles.includes(vehicle.id);
          const isFavorited = favoriteVehicles.includes(vehicle.id);

          return (
            <Card
              key={vehicle.id}
              className={`group relative bg-white/80 backdrop-blur-sm border-2 transition-all duration-300 cursor-pointer hover:shadow-2xl hover:shadow-purple-500/20 hover:-translate-y-2 ${
                isSelected
                  ? 'border-purple-500 shadow-lg shadow-purple-500/25 bg-gradient-to-br from-purple-50 to-pink-50'
                  : 'border-purple-200/50 hover:border-purple-300'
              }`}
              onClick={() => handleVehicleClick(vehicle.id)}
            >
              <div className="absolute -top-2 -right-2 z-10">
                <div className={`bg-gradient-to-r ${getMatchScoreColor(vehicle.match_score)} text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg`}>
                  {vehicle.match_score}% 매칭
                </div>
              </div>

              <button
                onClick={(e) => toggleFavorite(vehicle.id, e)}
                className={`absolute top-4 left-4 z-10 p-2 rounded-full backdrop-blur-sm transition-all duration-200 ${
                  isFavorited
                    ? 'bg-red-500 text-white shadow-lg'
                    : 'bg-white/70 text-gray-600 hover:bg-red-50 hover:text-red-500'
                }`}
              >
                <Heart className={`w-4 h-4 ${isFavorited ? 'fill-current' : ''}`} />
              </button>

              <div className="relative h-48 bg-gradient-to-br from-purple-100 to-pink-100 rounded-t-lg overflow-hidden">
                {vehicle.images && vehicle.images.length > 0 ? (
                  <img
                    src={vehicle.images[0]}
                    alt={`${vehicle.brand} ${vehicle.model}`}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Car className="w-16 h-16 text-purple-400" />
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              </div>

              <CardHeader className="pb-4">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <CardTitle className="text-xl font-bold text-gray-900 mb-1">
                      {vehicle.brand} {vehicle.model}
                    </CardTitle>
                    <div className="flex items-center gap-2 text-gray-600">
                      <Calendar className="w-4 h-4" />
                      <span>{vehicle.year}년</span>
                      <span className="text-gray-400">•</span>
                      <span>{vehicle.mileage?.toLocaleString()}km</span>
                    </div>
                  </div>
                </div>

                <div className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {vehicle.price?.toLocaleString()}만원
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${fuelInfo.color}`}>
                      <span className="text-sm">{fuelInfo.emoji}</span>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">연료</div>
                      <div className="text-sm font-medium">{fuelInfo.label}</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-orange-100 text-orange-800">
                      <Fuel className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">연비</div>
                      <div className="text-sm font-medium">{vehicle.fuel_efficiency}km/L</div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-600">{vehicle.location}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-gray-400" />
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <Star
                          key={i}
                          className={`w-3 h-3 ${
                            i < vehicle.safety_rating
                              ? 'text-yellow-400 fill-current'
                              : 'text-gray-300'
                          }`}
                        />
                      ))}
                      <span className="text-xs text-gray-500 ml-1">안전</span>
                    </div>
                  </div>
                </div>

                {vehicle.features && vehicle.features.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-sm font-medium text-gray-700">주요 옵션</div>
                    <div className="flex flex-wrap gap-1">
                      {vehicle.features.slice(0, 3).map((feature, index) => (
                        <Badge
                          key={index}
                          variant="secondary"
                          className="text-xs bg-purple-50 text-purple-700 border-purple-200"
                        >
                          {feature}
                        </Badge>
                      ))}
                      {vehicle.features.length > 3 && (
                        <Badge variant="outline" className="text-xs text-gray-500">
                          +{vehicle.features.length - 3}개
                        </Badge>
                      )}
                    </div>
                  </div>
                )}

                {vehicle.description && (
                  <p className="text-sm text-gray-600 line-clamp-2">
                    {vehicle.description}
                  </p>
                )}

                <div className="grid grid-cols-2 gap-2 pt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleVehicleClick(vehicle.id);
                    }}
                    className="border-purple-200 text-purple-600 hover:bg-purple-50 hover:border-purple-300"
                  >
                    <GitCompare className="w-4 h-4 mr-1" />
                    {isSelected ? '선택됨' : '선택'}
                  </Button>

                  <Button
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRequestFinancing(vehicle.id);
                    }}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white"
                  >
                    <CreditCard className="w-4 h-4 mr-1" />
                    대출
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {selectedVehicles.length > 0 && (
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200/50 text-center">
          <h3 className="text-lg font-semibold text-purple-800 mb-2">
            {selectedVehicles.length}/3 차량 선택됨
          </h3>
          <p className="text-purple-600 text-sm">
            {selectedVehicles.length === 3
              ? '3대 모두 선택되었습니다! 상세 비교 분석을 확인해보세요.'
              : `${3 - selectedVehicles.length}대 더 선택하시면 상세 비교 분석을 받을 수 있습니다.`
            }
          </p>
        </div>
      )}

      {vehicles.length === 0 && (
        <div className="text-center py-12">
          <Car className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            조건에 맞는 차량이 없습니다
          </h3>
          <p className="text-gray-500">
            검색 조건을 조정해보시거나 다른 옵션을 시도해보세요.
          </p>
        </div>
      )}
    </div>
  );
}